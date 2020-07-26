from PIL import Image

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    print(img)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

print("Training Running")
for i in range(0, 500):
    resizeImage("dataset/training/batu/batu" + str(i) + '.png')
    resizeImage("dataset/training/gunting/gunting" + str(i) + '.png')
    resizeImage("dataset/training/kertas/kertas" + str(i) + '.png')

print("Validation Running")
for i in range(0, 50):
    resizeImage("dataset/validation/batu/batu" + str(i) + '.png')
    resizeImage("dataset/validation/gunting/gunting" + str(i) + '.png')
    resizeImage("dataset/validation/kertas/kertas" + str(i) + '.png')


