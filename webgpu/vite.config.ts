import basicSsl from '@vitejs/plugin-basic-ssl'

export default {
    appType: 'mpa',
    plugins: [
        basicSsl()
    ]
}
