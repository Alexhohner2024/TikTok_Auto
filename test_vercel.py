import httpx
image_path = r"C:\Users\Babaian2023\Documents\MEGA\С Т Р А Х О В А Н И Е\Т А С\2026\Реклама\Видео Тикток\726358076_18327667018249965_6871118423391845148_n.jpg"
with open(image_path, "rb") as f:
    r = httpx.post("https://vercel-clean-image.vercel.app/", files={"file": f}, timeout=120.0)
print(f"Status: {r.status_code}")
print(f"Content-Type: {r.headers.get('content-type')}")
print(f"Size: {len(r.content)} bytes")
if r.status_code == 200:
    with open("vercel_test_output.jpg", "wb") as f:
        f.write(r.content)
    print("Saved to vercel_test_output.jpg")
else:
    print(r.text[:500])
