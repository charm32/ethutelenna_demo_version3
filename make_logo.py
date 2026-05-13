import base64

with open('eThute_Lenna_logo.jpg', 'rb') as f:
    data = base64.b64encode(f.read()).decode()

with open('logo_data.py', 'w') as out:
    out.write('LOGO_BASE64 = "' + data + '"\n')

print('Done! logo_data.py created successfully.')
