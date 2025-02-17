import os
from pathlib import Path

import numpy as np
from vulkan import *
import glfw

from utils import load_splat_file

validationLayers = ["VK_LAYER_KHRONOS_validation"]
deviceExtensions = [VK_KHR_SWAPCHAIN_EXTENSION_NAME]

enableValidationLayers = True


def debugCallback(*args):
    print('DEBUG: {} {}'.format(args[5], args[6]))
    return 0

def createDebugReportCallbackEXT(instance, pCreateInfo, pAllocator):
    func = vkGetInstanceProcAddr(instance, 'vkCreateDebugReportCallbackEXT')
    if func:
        return func(instance, pCreateInfo, pAllocator)
    else:
        return VK_ERROR_EXTENSION_NOT_PRESENT

def destroyDebugReportCallbackEXT(instance, callback, pAllocator):
    func = vkGetInstanceProcAddr(instance, 'vkDestroyDebugReportCallbackEXT')
    if func:
        func(instance, callback, pAllocator)

def destroySurface(instance, surface, pAllocator=None):
    func = vkGetInstanceProcAddr(instance, 'vkDestroySurfaceKHR')
    if func:
        func(instance, surface, pAllocator)

def destroySwapChain(device, swapChain, pAllocator=None):
    func = vkGetDeviceProcAddr(device, 'vkDestroySwapchainKHR')
    if func:
        func(device, swapChain, pAllocator)


class QueueFamilyIndices(object):

    def __init__(self):
        self.graphicsFamily = -1
        self.presentFamily = -1

    def isComplete(self):
        return self.graphicsFamily >= 0 and self.presentFamily >= 0


class SwapChainSupportDetails(object):
    def __init__(self):
        self.capabilities = None
        self.formats = None
        self.presentModes = None

class SplatsRendererVkGeo(object):

    def __init__(self, splat_file_path: str, width: int, height: int):
        self.width, self.height = width, height

        self.__window = None
        self.__instance = None
        self.__callback = None
        self.__surface = None
        self.__physicalDevice = None
        self.__device = None
        self.__graphicsQueue = None
        self.__presentQueue = None

        self.__swapChain = None
        self.__swapChainImages = None
        self.__swapChainImageFormat = None
        self.__swapChainExtent = None

        self.__swapChainImageViews = None
        self.__swapChainFramebuffers = None

        self.__renderPass = None
        self.__pipelineLayout = None
        self.__graphicsPipeline = None

        self.__commandPool = None
        self.__commandBuffers = None

        self.__imageAvailableSemaphore = None
        self.__renderFinishedSemaphore = None

        self.stride = 4 * (3 + 3 + 4 + 4)  # pos(3) + scale(3) + rot(4) + color(4)
        self.points = load_splat_file(splat_file_path)
        self.positions, self.scales, self.rotations, self.colors = self.points
        self.num_points = len(self.positions)
        self.vertex_buffer = None
        self.vertex_buffer_memory = None
        self.uniform_buffer = None
        self.uniform_buffer_memory = None
        self.descriptor_pool = None
        self.descriptor_set_layout = None
        self.descriptor_sets = None
        self.view_proj_ubo = None

        self.__initWindow()
        self.__initVulkan()


    def sort(self, viewProj):
        if viewProj is None:
            indices = np.arange(len(self.positions), dtype=np.uint32)
        else:
            positions_v4 = np.hstack([self.positions, np.ones((len(self.positions), 1))])
            cam = (viewProj @ positions_v4.T).T
            depths = cam[:, 2]
            indices = np.argsort(depths).astype(np.uint32)

        # indices = indices[::-1]

        # TODO upload order into gpu instead later
        self.positions = self.positions[indices]
        self.scales = self.scales[indices]
        self.rotations = self.rotations[indices]
        self.colors = self.colors[indices]
        self.__createVertexBuffer()


    def draw(self, view, proj, w, h, f):
        # Update the UBO with the new view and projection matrices

        self.view_proj_ubo['view'] = view
        self.view_proj_ubo['proj'] = proj
        self.view_proj_ubo['width'] = w
        self.view_proj_ubo['height'] = h
        self.view_proj_ubo['focal'] = f

        # Copy the data to the mapped memory
        data_ptr = vkMapMemory(self.__device, self.uniform_buffer_memory, 0, self.view_proj_ubo.itemsize, 0)
        ffi.memmove(data_ptr, ffi.from_buffer(self.view_proj_ubo), self.view_proj_ubo.itemsize)
        vkUnmapMemory(self.__device, self.uniform_buffer_memory)


    def __del__(self):
        vkDeviceWaitIdle(self.__device)

        # --- Cleanup Point Cloud Specific Resources ---
        if self.vertex_buffer:
            vkDestroyBuffer(self.__device, self.vertex_buffer, None)
        if self.vertex_buffer_memory:
            vkFreeMemory(self.__device, self.vertex_buffer_memory, None)

        if self.uniform_buffer:
            vkDestroyBuffer(self.__device, self.uniform_buffer, None)
        if self.uniform_buffer_memory:
            vkFreeMemory(self.__device, self.uniform_buffer_memory, None)

        if self.descriptor_pool:
            vkDestroyDescriptorPool(self.__device, self.descriptor_pool, None)

        if self.descriptor_set_layout:
            vkDestroyDescriptorSetLayout(self.__device, self.descriptor_set_layout, None)


        if self.__imageAvailableSemaphore:
            vkDestroySemaphore(self.__device, self.__imageAvailableSemaphore, None)

        if self.__renderFinishedSemaphore:
            vkDestroySemaphore(self.__device, self.__renderFinishedSemaphore, None)

        if self.__commandBuffers:
            self.__commandBuffers = None

        if self.__commandPool:
            vkDestroyCommandPool(self.__device, self.__commandPool, None)

        if self.__swapChainFramebuffers:
            for i in self.__swapChainFramebuffers:
                vkDestroyFramebuffer(self.__device, i, None)
            self.__swapChainFramebuffers = None

        if self.__renderPass:
            vkDestroyRenderPass(self.__device, self.__renderPass, None)

        if self.__pipelineLayout:
            vkDestroyPipelineLayout(self.__device, self.__pipelineLayout, None)

        if self.__graphicsPipeline:
            vkDestroyPipeline(self.__device, self.__graphicsPipeline, None)

        if self.__swapChainImageViews:
            for i in self.__swapChainImageViews:
                vkDestroyImageView(self.__device, i, None)

        if self.__swapChain:
            destroySwapChain(self.__device, self.__swapChain, None)

        if self.__device:
            vkDestroyDevice(self.__device, None)

        if self.__surface:
            destroySurface(self.__instance, self.__surface, None)

        if self.__callback:
            destroyDebugReportCallbackEXT(self.__instance, self.__callback, None)

        if self.__instance:
            vkDestroyInstance(self.__instance, None)

    def __initWindow(self):
        glfw.init()

        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, False)

        self.__window = glfw.create_window(self.width, self.height, "Vulkan", None, None)

    def __initVulkan(self):
        self.__createInstance()
        self.__setupDebugCallback()
        self.__createSurface()
        self.__pickPhysicalDevice()
        self.__createLogicalDevice()
        self.__createSwapChain()
        self.__createImageViews()
        self.__createRenderPass()
        self.__createDescriptorSetLayout()  # Before pipeline
        self.__createGraphicsPipeline()
        self.__createFramebuffers()
        self.__createCommandPool()
        self.__createVertexBuffer()        # Point cloud vertex data
        self.__createUniformBuffer()         # For view/proj matrices
        self.__createDescriptorPool()
        self.__createDescriptorSets()
        self.__createCommandBuffers()        # Now includes point cloud drawing
        self.__createSemaphores()

    def __createInstance(self):
        if enableValidationLayers and not self.__checkValidationLayerSupport():
            raise Exception("validation layers requested, but not available!")

        appInfo = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName='Hello Triangle',
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName='No Engine',
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_MAKE_VERSION(1, 0, 3)
        )

        createInfo = None
        extensions = self.__getRequiredExtensions()

        if enableValidationLayers:
            createInfo = VkInstanceCreateInfo(
                sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pApplicationInfo=appInfo,
                enabledExtensionCount=len(extensions),
                ppEnabledExtensionNames=extensions,
                enabledLayerCount=len(validationLayers),
                ppEnabledLayerNames=validationLayers
            )
        else:
            createInfo = VkInstanceCreateInfo(
                sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pApplicationInfo=appInfo,
                enabledExtensionCount=len(extensions),
                ppEnabledExtensionNames=extensions,
                enabledLayerCount=0
            )

        self.__instance = vkCreateInstance(createInfo, None)

    def __setupDebugCallback(self):
        if not enableValidationLayers:
            return

        createInfo = VkDebugReportCallbackCreateInfoEXT(
            sType=VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
            flags=VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT,
            pfnCallback=debugCallback
        )
        self.__callback = createDebugReportCallbackEXT(self.__instance, createInfo, None)
        if not self.__callback:
            raise Exception("failed to set up debug callback!")

    def __createSurface(self):
        surface_ptr = ffi.new('VkSurfaceKHR[1]')
        glfw.create_window_surface(self.__instance, self.__window, None, surface_ptr)
        self.__surface = surface_ptr[0]
        if self.__surface is None:
            raise Exception("failed to create window surface!")

    def __pickPhysicalDevice(self):
        devices = vkEnumeratePhysicalDevices(self.__instance)

        for device in devices:
            if self.__isDeviceSuitable(device):
                self.__physicalDevice = device
                break

        if self.__physicalDevice is None:
            raise Exception("failed to find a suitable GPU!")

    def __createLogicalDevice(self):
        indices = self.__findQueueFamilies(self.__physicalDevice)
        uniqueQueueFamilies = {}.fromkeys((indices.graphicsFamily, indices.presentFamily))
        queueCreateInfos = []
        for queueFamily in uniqueQueueFamilies:
            queueCreateInfo = VkDeviceQueueCreateInfo(
                sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=queueFamily,
                queueCount=1,
                pQueuePriorities=[1.0]
            )
            queueCreateInfos.append(queueCreateInfo)

        deviceFeatures = VkPhysicalDeviceFeatures()
        createInfo = None
        if enableValidationLayers:
            createInfo = VkDeviceCreateInfo(
                sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                flags=0,
                pQueueCreateInfos=queueCreateInfos,
                queueCreateInfoCount=len(queueCreateInfos),
                pEnabledFeatures=[deviceFeatures],
                enabledExtensionCount=len(deviceExtensions),
                ppEnabledExtensionNames=deviceExtensions,
                enabledLayerCount=len(validationLayers),
                ppEnabledLayerNames=validationLayers
            )
        else:
            createInfo = VkDeviceCreateInfo(
                sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                flags=0,
                pQueueCreateInfos=queueCreateInfos,
                queueCreateInfoCount=len(queueCreateInfos),
                pEnabledFeatures=[deviceFeatures],
                enabledExtensionCount=len(deviceExtensions),
                ppEnabledExtensionNames=deviceExtensions,
                enabledLayerCount=0
            )

        self.__device = vkCreateDevice(self.__physicalDevice, createInfo, None)
        if self.__device is None:
            raise Exception("failed to create logical device!")
        self.__graphicsQueue = vkGetDeviceQueue(self.__device, indices.graphicsFamily, 0)
        self.__presentQueue = vkGetDeviceQueue(self.__device, indices.presentFamily, 0)

    def __createSwapChain(self):
        swapChainSupport = self.__querySwapChainSupport(self.__physicalDevice)

        surfaceFormat = self.__chooseSwapSurfaceFormat(swapChainSupport.formats)
        presentMode = self.__chooseSwapPresentMode(swapChainSupport.presentModes)
        extent = self.__chooseSwapExtent(swapChainSupport.capabilities)

        imageCount = swapChainSupport.capabilities.minImageCount + 1
        if swapChainSupport.capabilities.maxImageCount > 0 and imageCount > swapChainSupport.capabilities.maxImageCount:
            imageCount = swapChainSupport.capabilities.maxImageCount

        createInfo = VkSwapchainCreateInfoKHR(
            sType=VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            flags=0,
            surface=self.__surface,
            minImageCount=imageCount,
            imageFormat=surfaceFormat.format,
            imageColorSpace=surfaceFormat.colorSpace,
            imageExtent=extent,
            imageArrayLayers=1,
            imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
        )

        indices = self.__findQueueFamilies(self.__physicalDevice)
        if indices.graphicsFamily != indices.presentFamily:
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT
            createInfo.queueFamilyIndexCount = 2
            createInfo.pQueueFamilyIndices = [indices.graphicsFamily, indices.presentFamily]
        else:
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
        createInfo.presentMode = presentMode
        createInfo.clipped = True

        vkCreateSwapchainKHR = vkGetDeviceProcAddr(self.__device, 'vkCreateSwapchainKHR')
        self.__swapChain = vkCreateSwapchainKHR(self.__device, createInfo, None)

        vkGetSwapchainImagesKHR = vkGetDeviceProcAddr(self.__device, 'vkGetSwapchainImagesKHR')
        self.__swapChainImages = vkGetSwapchainImagesKHR(self.__device, self.__swapChain)

        self.__swapChainImageFormat = surfaceFormat.format
        self.__swapChainExtent = extent

    def __createImageViews(self):
        self.__swapChainImageViews = []
        components = VkComponentMapping(VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY)
        subresourceRange = VkImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT,
                                                   0, 1, 0, 1)
        for i, image in enumerate(self.__swapChainImages):
            createInfo = VkImageViewCreateInfo(
                sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                flags=0,
                image=image,
                viewType=VK_IMAGE_VIEW_TYPE_2D,
                format=self.__swapChainImageFormat,
                components=components,
                subresourceRange=subresourceRange
            )
            self.__swapChainImageViews.append(vkCreateImageView(self.__device, createInfo, None))

    def __createRenderPass(self):
        colorAttachment = VkAttachmentDescription(
            format=self.__swapChainImageFormat,
            samples=VK_SAMPLE_COUNT_1_BIT,
            loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        )

        colorAttachmentRef = VkAttachmentReference(
            attachment=0,
            layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )

        subPass = VkSubpassDescription(
            pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=colorAttachmentRef
        )

        renderPassInfo = VkRenderPassCreateInfo(
            sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount=1,
            pAttachments=colorAttachment,
            subpassCount=1,
            pSubpasses=subPass
        )

        self.__renderPass = vkCreateRenderPass(self.__device, renderPassInfo, None)

    def __createDescriptorSetLayout(self):
        uboLayoutBinding = VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            stageFlags=VK_SHADER_STAGE_VERTEX_BIT,
            pImmutableSamplers=None
        )

        layoutInfo = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=1,
            pBindings=uboLayoutBinding  # Corrected: Pass the binding directly
        )

        self.descriptor_set_layout = vkCreateDescriptorSetLayout(self.__device, layoutInfo, None)
        if self.descriptor_set_layout is None:
            raise RuntimeError("Failed to create descriptor set layout!")


    def __createGraphicsPipeline(self):
        dir = Path(__file__).resolve().parent / 'vk_shaders'

        vertShaderModule = self.__createShaderModule(dir / 'points_vert.spv')
        fragShaderModule = self.__createShaderModule(dir / 'points_frag.spv')

        vertShaderStageInfo = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            flags=0,
            stage=VK_SHADER_STAGE_VERTEX_BIT,
            module=vertShaderModule,
            pName='main'
        )

        fragShaderStageInfo = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            flags=0,
            stage=VK_SHADER_STAGE_FRAGMENT_BIT,
            module=fragShaderModule,
            pName='main'
        )

        shaderStages = [vertShaderStageInfo, fragShaderStageInfo]

        # --- Vertex Input Description ---
        bindingDescription = VkVertexInputBindingDescription(
            binding=0,
            stride=self.stride,
            inputRate=VK_VERTEX_INPUT_RATE_VERTEX
        )

        offset = 0
        positionAttributeDescription = VkVertexInputAttributeDescription(
            binding=0,
            location=0,
            format=VK_FORMAT_R32G32B32_SFLOAT,  # XYZ position
            offset=offset
        )
        offset += 3 * 4

        scaleAttributeDescription = VkVertexInputAttributeDescription(
            binding=0,
            location=1,
            format=VK_FORMAT_R32G32B32_SFLOAT,  # scale
            offset=offset
        )
        offset += 3 * 4

        rotAttributeDescription = VkVertexInputAttributeDescription(
            binding = 0,
            location = 2,
            format = VK_FORMAT_R32G32B32A32_SFLOAT,  # quaternion
            offset = offset
        )
        offset += 4 * 4

        colorAttributeDescription = VkVertexInputAttributeDescription(
            binding=0,
            location=3,
            format=VK_FORMAT_R32G32B32A32_SFLOAT,  # color (RGBA)
            offset=offset
        )
        offset += 4 * 4

        assert offset == self.stride

        vertexInputInfo = VkPipelineVertexInputStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount=1,
            pVertexBindingDescriptions=bindingDescription,
            vertexAttributeDescriptionCount=4,
            pVertexAttributeDescriptions=[positionAttributeDescription, scaleAttributeDescription, rotAttributeDescription, colorAttributeDescription]
        )
        # --- Input Assembly ---
        inputAssembly = VkPipelineInputAssemblyStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology=VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
            primitiveRestartEnable=False # False for points
        )

        viewport = VkViewport(0.0, 0.0,
                              float(self.__swapChainExtent.width), float(self.__swapChainExtent.height),
                              0.0, 1.0)
        scissor = VkRect2D([0, 0], self.__swapChainExtent)
        viewportState = VkPipelineViewportStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount=1,
            pViewports=viewport,
            scissorCount=1,
            pScissors=scissor
        )

        rasterizer = VkPipelineRasterizationStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depthClampEnable=False,
            rasterizerDiscardEnable=False,
            polygonMode=VK_POLYGON_MODE_FILL,
            lineWidth=1.0,
            cullMode=VK_CULL_MODE_BACK_BIT,
            frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE, # counter clock wise
            depthBiasEnable=False
        )

        multisampling = VkPipelineMultisampleStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            sampleShadingEnable=False,
            rasterizationSamples=VK_SAMPLE_COUNT_1_BIT
        )

        colorBlendAttachment = VkPipelineColorBlendAttachmentState(
            colorWriteMask=VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            blendEnable=True,
            srcColorBlendFactor=VK_BLEND_FACTOR_SRC_ALPHA,
            dstColorBlendFactor=VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            colorBlendOp=VK_BLEND_OP_ADD,
            srcAlphaBlendFactor=VK_BLEND_FACTOR_ONE,
            dstAlphaBlendFactor=VK_BLEND_FACTOR_ZERO,
            alphaBlendOp=VK_BLEND_OP_ADD
        )

        colorBlending = VkPipelineColorBlendStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logicOpEnable=False,
            logicOp=VK_LOGIC_OP_COPY,
            attachmentCount=1,
            pAttachments=colorBlendAttachment,
            blendConstants=[0.0, 0.0, 0.0, 0.0]
        )

        # --- Pipeline Layout (with descriptor set layout) ---
        set_layouts = ffi.new('VkDescriptorSetLayout[]', [self.descriptor_set_layout]) # Corrected
        pipelineLayoutInfo = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,  #  One descriptor set layout
            pSetLayouts=set_layouts, # Pass the pointer to the C array
            pushConstantRangeCount=0
        )

        self.__pipelineLayout = vkCreatePipelineLayout(self.__device, pipelineLayoutInfo, None)
        if self.__pipelineLayout is None:
            raise RuntimeError("Failed to create pipeline layout!")


        # --- Graphics Pipeline ---
        pipelineInfo = VkGraphicsPipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            stageCount=2,
            pStages=shaderStages,
            pVertexInputState=vertexInputInfo,
            pInputAssemblyState=inputAssembly,
            pViewportState=viewportState,
            pRasterizationState=rasterizer,
            pMultisampleState=multisampling,
            pColorBlendState=colorBlending,
            layout=self.__pipelineLayout,
            renderPass=self.__renderPass,
            subpass=0
        )

        self.__graphicsPipeline = vkCreateGraphicsPipelines(self.__device, VK_NULL_HANDLE, 1, pipelineInfo, None)[0]
        if self.__graphicsPipeline is None:
             raise RuntimeError("Failed to create graphics pipeline!")


        vkDestroyShaderModule(self.__device, vertShaderModule, None)
        vkDestroyShaderModule(self.__device, fragShaderModule, None)

    def __createFramebuffers(self):
        self.__swapChainFramebuffers = []

        for imageView in self.__swapChainImageViews:
            attachments = [imageView,]

            framebufferInfo = VkFramebufferCreateInfo(
                sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                renderPass=self.__renderPass,
                attachmentCount=1,
                pAttachments=attachments,
                width=self.__swapChainExtent.width,
                height=self.__swapChainExtent.height,
                layers=1
            )
            framebuffer = vkCreateFramebuffer(self.__device, framebufferInfo, None)
            self.__swapChainFramebuffers.append(framebuffer)

    def __createCommandPool(self):
        queueFamilyIndices = self.__findQueueFamilies(self.__physicalDevice)

        poolInfo = VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=queueFamilyIndices.graphicsFamily
        )

        self.__commandPool = vkCreateCommandPool(self.__device, poolInfo, None)

    def __createVertexBuffer(self):
        bufferSize = self.stride * self.num_points

        stagingBuffer, stagingBufferMemory = self.__createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

        # Combine position and color data into a single array
        vertex_data = np.column_stack([self.positions, self.scales, self.rotations, self.colors]).flatten().astype(np.float32)

        data_ptr = vkMapMemory(self.__device, stagingBufferMemory, 0, bufferSize, 0)
        ffi.memmove(data_ptr, vertex_data.data, bufferSize) # vertex_data.data is nsc.Array and is contiguous
        vkUnmapMemory(self.__device, stagingBufferMemory)

        self.vertex_buffer, self.vertex_buffer_memory = self.__createBuffer(bufferSize,
                                                                              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

        self.__copyBuffer(stagingBuffer, self.vertex_buffer, bufferSize)

        vkDestroyBuffer(self.__device, stagingBuffer, None)
        vkFreeMemory(self.__device, stagingBufferMemory, None)

    def __createUniformBuffer(self):
        self.view_proj_ubo = np.zeros(1, dtype=[
            ('view', np.float32, (4, 4)),
            ('proj', np.float32, (4, 4)),
            ('width', np.float32, (1)),
            ('height', np.float32, (1)),
            ('focal', np.float32, (1))
        ])

        bufferSize = self.view_proj_ubo.itemsize
        self.uniform_buffer, self.uniform_buffer_memory = self.__createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

    def __createDescriptorPool(self):
        poolSize = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1
        )
        poolInfo = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=1,
            pPoolSizes=poolSize,
            maxSets=1  #  One set for our UBO
        )

        self.descriptor_pool = vkCreateDescriptorPool(self.__device, poolInfo, None)


    def __createDescriptorSets(self):

        layouts = [self.descriptor_set_layout] #, ] * len(self.__swapChainImages)
        allocInfo = VkDescriptorSetAllocateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=1,  # One set per swapchain image
            pSetLayouts=layouts
        )
        descriptor_sets = vkAllocateDescriptorSets(self.__device, allocInfo)
        self.descriptor_sets = [ffi.addressof(descriptor_sets, i)[0] for i in range(1)]

        descriptorBufferInfo = VkDescriptorBufferInfo(
            buffer=self.uniform_buffer,
            offset=0,
            range=self.view_proj_ubo.itemsize  # Or VK_WHOLE_SIZE
        )
        descriptorWrite = VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.descriptor_sets[0],
            dstBinding=0,
            dstArrayElement=0,
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            pBufferInfo=descriptorBufferInfo
        )

        vkUpdateDescriptorSets(self.__device, 1, descriptorWrite, 0, None)


    def __createCommandBuffers(self):
        # self.__commandBuffers = []

        allocInfo = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.__commandPool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=len(self.__swapChainFramebuffers)
        )

        commandBuffers = vkAllocateCommandBuffers(self.__device, allocInfo)
        self.__commandBuffers = [ffi.addressof(commandBuffers, i)[0] for i in range(len(self.__swapChainFramebuffers))]

        for i, cmdBuffer in enumerate(self.__commandBuffers):
            beginInfo = VkCommandBufferBeginInfo(
                sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
            )

            vkBeginCommandBuffer(cmdBuffer, beginInfo)

            renderPassInfo = VkRenderPassBeginInfo(
                sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                renderPass=self.__renderPass,
                framebuffer=self.__swapChainFramebuffers[i],
                renderArea=[[0, 0], self.__swapChainExtent]
            )

            clearColor = VkClearValue([[0.0, 0.0, 0.0, 1.0]])
            renderPassInfo.clearValueCount = 1
            renderPassInfo.pClearValues = ffi.addressof(clearColor)

            vkCmdBeginRenderPass(cmdBuffer, renderPassInfo, VK_SUBPASS_CONTENTS_INLINE)

            vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, self.__graphicsPipeline)

            # --- Bind Vertex Buffer ---
            vkCmdBindVertexBuffers(cmdBuffer, 0, 1, [self.vertex_buffer, ], [0, ])

            # --- Bind Descriptor Set (UBO) ---
            vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, self.__pipelineLayout, 0, 1, self.descriptor_sets, 0, None)

            # --- Draw Points ---
            vkCmdDraw(cmdBuffer, self.num_points, 1, 0, 0)  # Draw all points

            vkCmdEndRenderPass(cmdBuffer)

            vkEndCommandBuffer(cmdBuffer)

    def __createSemaphores(self):
        semaphoreInfo = VkSemaphoreCreateInfo(sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO)

        self.__imageAvailableSemaphore = vkCreateSemaphore(self.__device, semaphoreInfo, None)
        self.__renderFinishedSemaphore = vkCreateSemaphore(self.__device, semaphoreInfo, None)

    def __createBuffer(self, size, usage, properties):
        bufferInfo = VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE
        )

        buffer = vkCreateBuffer(self.__device, bufferInfo, None)

        memRequirements = vkGetBufferMemoryRequirements(self.__device, buffer)

        allocInfo = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=memRequirements.size,
            memoryTypeIndex=self.__findMemoryType(memRequirements.memoryTypeBits, properties)
        )

        memory = vkAllocateMemory(self.__device, allocInfo, None)

        vkBindBufferMemory(self.__device, buffer, memory, 0)

        return buffer, memory

    def __findMemoryType(self, typeFilter, properties):
        memProperties = vkGetPhysicalDeviceMemoryProperties(self.__physicalDevice)

        for i, memoryType in enumerate(memProperties.memoryTypes):
            if (typeFilter & (1 << i)) and (memoryType.propertyFlags & properties) == properties:
                return i

        raise Exception("failed to find suitable memory type!")

    def __copyBuffer(self, srcBuffer, dstBuffer, size):
        allocInfo = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.__commandPool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )

        commandBuffers = vkAllocateCommandBuffers(self.__device, allocInfo)
        commandBuffer = ffi.addressof(commandBuffers, 0)[0]

        beginInfo = VkCommandBufferBeginInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )

        vkBeginCommandBuffer(commandBuffer, beginInfo)

        copyRegion = VkBufferCopy(0, 0, size)
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, copyRegion)

        vkEndCommandBuffer(commandBuffer)

        submitInfo = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[commandBuffer]
        )

        vkQueueSubmit(self.__graphicsQueue, 1, submitInfo, VK_NULL_HANDLE)
        vkQueueWaitIdle(self.__graphicsQueue)  # TODO use a fence to avoid waiting if possible

        vkFreeCommandBuffers(self.__device, self.__commandPool, 1, [commandBuffer])

    def __drawFrame(self):
        vkAcquireNextImageKHR = vkGetDeviceProcAddr(self.__device, 'vkAcquireNextImageKHR')
        vkQueuePresentKHR = vkGetDeviceProcAddr(self.__device, 'vkQueuePresentKHR')

        imageIndex = vkAcquireNextImageKHR(self.__device, self.__swapChain, 18446744073709551615,
                                           self.__imageAvailableSemaphore, VK_NULL_HANDLE)

        submitInfo = VkSubmitInfo(sType=VK_STRUCTURE_TYPE_SUBMIT_INFO)

        waitSemaphores = ffi.new('VkSemaphore[]', [self.__imageAvailableSemaphore])
        waitStages = ffi.new('uint32_t[]', [VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, ])
        submitInfo.waitSemaphoreCount = 1
        submitInfo.pWaitSemaphores = waitSemaphores
        submitInfo.pWaitDstStageMask = waitStages

        cmdBuffers = ffi.new('VkCommandBuffer[]', [self.__commandBuffers[imageIndex], ])
        submitInfo.commandBufferCount = 1
        submitInfo.pCommandBuffers = cmdBuffers

        signalSemaphores = ffi.new('VkSemaphore[]', [self.__renderFinishedSemaphore])
        submitInfo.signalSemaphoreCount = 1
        submitInfo.pSignalSemaphores = signalSemaphores

        vkQueueSubmit(self.__graphicsQueue, 1, submitInfo, VK_NULL_HANDLE)

        swapChains = [self.__swapChain]
        presentInfo = VkPresentInfoKHR(
            sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            waitSemaphoreCount=1,
            pWaitSemaphores=signalSemaphores,
            swapchainCount=1,
            pSwapchains=swapChains,
            pImageIndices=[imageIndex]
        )

        vkQueuePresentKHR(self.__presentQueue, presentInfo)

    def __createShaderModule(self, shaderFile):
        with open(shaderFile, 'rb') as sf:
            code = sf.read()
            codeSize = len(code)

            createInfo = VkShaderModuleCreateInfo(
                sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=codeSize,
                pCode=code
            )

            return vkCreateShaderModule(self.__device, createInfo, None)

    def __chooseSwapSurfaceFormat(self, availableFormats):
        if len(availableFormats) == 1 and availableFormats[0].format == VK_FORMAT_UNDEFINED:
            return VkSurfaceFormatKHR(VK_FORMAT_B8G8R8A8_UNORM, 0)

        for availableFormat in availableFormats:
            if availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM and availableFormat.colorSpace == 0:
                return availableFormat

        return availableFormats[0]

    def __chooseSwapPresentMode(self, availablePresentModes):
        for availablePresentMode in availablePresentModes:
            if availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR:
                return availablePresentMode

        return VK_PRESENT_MODE_FIFO_KHR

    def __chooseSwapExtent(self, capabilities):
        width = max(capabilities.minImageExtent.width, min(capabilities.maxImageExtent.width, self.width))
        height = max(capabilities.minImageExtent.height, min(capabilities.maxImageExtent.height, self.height))
        return VkExtent2D(width, height)

    def __querySwapChainSupport(self, device):
        details = SwapChainSupportDetails()

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR = vkGetInstanceProcAddr(self.__instance,
                                                                          'vkGetPhysicalDeviceSurfaceCapabilitiesKHR')
        details.capabilities = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, self.__surface)

        vkGetPhysicalDeviceSurfaceFormatsKHR = vkGetInstanceProcAddr(self.__instance,
                                                                     'vkGetPhysicalDeviceSurfaceFormatsKHR')
        details.formats = vkGetPhysicalDeviceSurfaceFormatsKHR(device, self.__surface)

        vkGetPhysicalDeviceSurfacePresentModesKHR = vkGetInstanceProcAddr(self.__instance,
                                                                          'vkGetPhysicalDeviceSurfacePresentModesKHR')
        details.presentModes = vkGetPhysicalDeviceSurfacePresentModesKHR(device, self.__surface)

        return details

    def __isDeviceSuitable(self, device):
        indices = self.__findQueueFamilies(device)
        extensionsSupported = self.__checkDeviceExtensionSupport(device)
        swapChainAdequate = False
        if extensionsSupported:
            swapChainSupport = self.__querySwapChainSupport(device)
            swapChainAdequate = (not swapChainSupport.formats is None) and (
                not swapChainSupport.presentModes is None)
        return indices.isComplete() and extensionsSupported and swapChainAdequate

    def __checkDeviceExtensionSupport(self, device):
        availableExtensions = vkEnumerateDeviceExtensionProperties(device, None)

        for extension in availableExtensions:
            if extension.extensionName in deviceExtensions:
                return True

        return False

    def __findQueueFamilies(self, device):
        vkGetPhysicalDeviceSurfaceSupportKHR = vkGetInstanceProcAddr(self.__instance,
                                                                     'vkGetPhysicalDeviceSurfaceSupportKHR')
        indices = QueueFamilyIndices()

        queueFamilies = vkGetPhysicalDeviceQueueFamilyProperties(device)

        for i, queueFamily in enumerate(queueFamilies):
            if queueFamily.queueCount > 0 and queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT:
                indices.graphicsFamily = i

            presentSupport = vkGetPhysicalDeviceSurfaceSupportKHR(device, i, self.__surface)

            if queueFamily.queueCount > 0 and presentSupport:
                indices.presentFamily = i

            if indices.isComplete():
                break

        return indices

    def __getRequiredExtensions(self):
        extensions = list(map(str, glfw.get_required_instance_extensions()))

        if enableValidationLayers:
            extensions.append(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)

        return extensions

    def __checkValidationLayerSupport(self):
        availableLayers = vkEnumerateInstanceLayerProperties()
        for layerName in validationLayers:
            layerFound = False

            for layerProperties in availableLayers:
                if layerName == layerProperties.layerName:
                    layerFound = True
                    break
            if not layerFound:
                return False

        return True

    def loop(self, view, proj, w, h, f):
        while not glfw.window_should_close(self.__window):
            glfw.poll_events()
            self.draw(view, proj, w, h, f)
            self.__drawFrame()