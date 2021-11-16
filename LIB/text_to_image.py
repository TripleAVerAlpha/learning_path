from PIL import Image, ImageDraw, ImageFont


def text_to_image(text, file=None):
    lines = tuple(l.rstrip() for l in text.split("\n"))
    font = ImageFont.truetype('LIB/Font/Comic_CAT.otf', size=60)
    pt2px = lambda pt: int(round(pt * 96.0 / 72))  # convert points to pixels
    max_width_line = max(lines, key=lambda s: font.getsize(s)[0])
    # max height is adjusted down because it too large visually for spacing
    test_string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    max_height = pt2px(font.getsize(test_string)[1])
    max_width = pt2px(font.getsize(max_width_line)[0])
    height = max_height * len(lines)+20  # perfect or a little oversized
    width = int(round(max_width))+20  # a little oversized
    image = Image.new('RGB', (width, height), color='#0d1117')
    draw_text = ImageDraw.Draw(image)
    draw_text.text((10, 10), text, font=font, fill="#c9d1d9")
    image.show()
    if not (file is None):
        image.save(file)
    return image
