// Each vertex: position (4 floats) + color (4 floats) = 8 floats
export const cubeVertexSize = 4 * 8; // Byte size of one cube vertex (32 bytes)
export const cubePositionOffset = 0;
export const cubeColorOffset = 4 * 4;
export const cubeVertexCount = 8;

// prettier-ignore
export const cubeVertexArray = new Float32Array([
    // Position (x, y, z, w)   Color (r, g, b, a)
    -1, -1,  1, 1,   1, 0, 0, 1, // Red
    1, -1,  1, 1,   0, 1, 0, 1, // Green
    1,  1,  1, 1,   0, 0, 1, 1, // Blue
    -1,  1,  1, 1,   1, 1, 0, 1, // Yellow
    -1, -1, -1, 1,   0, 1, 1, 1, // Cyan
    1, -1, -1, 1,   1, 0, 1, 1, // Magenta
    1,  1, -1, 1,   1, 1, 1, 1, // White
    -1,  1, -1, 1,   0.2, 0.2, 0.2, 1, // Dark Gray
]);