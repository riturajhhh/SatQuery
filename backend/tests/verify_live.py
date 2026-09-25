import urllib.request
import json
import numpy as np
import io
import uuid
from PIL import Image

def main():
    # 1. Create a test image with 3 distinct buildings
    arr = np.zeros((160, 160, 3), dtype=np.uint8)
    arr[:, :, 0] = 45; arr[:, :, 1] = 135; arr[:, :, 2] = 50 # Green terrain
    arr[20:45, 20:45, :] = [210, 205, 195] # Building 1 (NW)
    arr[25:50, 95:130, :] = [190, 185, 180] # Building 2 (NE)
    arr[95:135, 95:135, :] = [225, 220, 215] # Building 3 (SE)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    file_bytes = buf.getvalue()

    boundary = '----WebKitFormBoundary' + uuid.uuid4().hex
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="files"; filename="test_sat.png"\r\n'
        f'Content-Type: image/png\r\n\r\n'
    ).encode('utf-8') + file_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')

    req = urllib.request.Request(
        'http://127.0.0.1:8000/api/upload',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
    )
    with urllib.request.urlopen(req) as resp:
        up_data = json.loads(resp.read().decode())

    upload_id = up_data['upload_id']
    print(f"UPLOAD ID: {upload_id}")

    # 2. Test Building Count Query
    req1 = urllib.request.Request(
        'http://127.0.0.1:8000/api/analyze',
        data=json.dumps({'upload_id': upload_id, 'query': 'How many buildings are in this satellite image?'}).encode(),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req1) as resp:
        a1_data = json.loads(resp.read().decode())
    a1_id = a1_data['analysis_id']

    with urllib.request.urlopen(f'http://127.0.0.1:8000/api/analysis/{a1_id}') as resp:
        res1 = json.loads(resp.read().decode())

    print('\n=== QUERY 1 (BUILDING COUNT) RESULT ===')
    print('Answer:', res1.get('answer', {}).get('text'))
    print('Confidence:', res1.get('answer', {}).get('confidence'))
    boxes = res1.get('evidence', {}).get('grounding_overlay', {}).get('boxes', [])
    print('Boxes count:', len(boxes))
    print('Overlay URL:', res1.get('evidence', {}).get('grounding_overlay', {}).get('overlay_url'))

    # 3. Test Land Cover Description Query
    req2 = urllib.request.Request(
        'http://127.0.0.1:8000/api/analyze',
        data=json.dumps({'upload_id': upload_id, 'query': 'Describe the land cover of this scene in detail.'}).encode(),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req2) as resp:
        a2_data = json.loads(resp.read().decode())
    a2_id = a2_data['analysis_id']

    with urllib.request.urlopen(f'http://127.0.0.1:8000/api/analysis/{a2_id}') as resp:
        res2 = json.loads(resp.read().decode())

    print('\n=== QUERY 2 (LAND COVER) RESULT ===')
    print('Answer:', res2.get('answer', {}).get('text'))
    print('Spectral Indices:', res2.get('evidence', {}).get('spectral_indices', {}).get('spatial_indices'))

if __name__ == '__main__':
    main()
