#@markdown ### Launch: ngrok tunnel + uvicorn server
import nest_asyncio, uvicorn
from pyngrok import ngrok, conf

try:
    from google.colab import userdata
    NGROK_AUTHTOKEN = userdata.get('NGROK_AUTHTOKEN')
except Exception:
    NGROK_AUTHTOKEN = None

if not NGROK_AUTHTOKEN:
    import getpass
    NGROK_AUTHTOKEN = getpass.getpass('Enter your ngrok authtoken (input hidden): ')

conf.get_default().auth_token = NGROK_AUTHTOKEN

ngrok.kill()  # clear any stale tunnels from a previous run
public_url = ngrok.connect(8000, "http")
print('=' * 55)
print(f'  🌍 Public URL: {public_url}')
print('=' * 55)
print('Open that link in a browser tab — that is your web app.')
print('Leave this cell running; stopping it shuts the server down.')

nest_asyncio.apply()
config = uvicorn.Config(app, host='0.0.0.0', port=8000)
server = uvicorn.Server(config)
await server.serve()
