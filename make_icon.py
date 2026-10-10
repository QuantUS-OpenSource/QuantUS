from PIL import Image

img = Image.open("Images/logo.png").resize((256, 256))

img.save("Images/logo.ico", format="ICO",
         sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])

print("Done — Images/logo.ico created")