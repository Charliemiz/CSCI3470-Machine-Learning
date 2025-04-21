from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt      # if you want the Matplotlib option

outdir = Path("gan_simple")          # this is the folder where samples were saved
SAVE_ITERS = {1, 200, 400, 600}

for it in sorted(SAVE_ITERS):
    img_path = outdir / f"samples_{it:04d}.png"
    if img_path.exists():
        # --- option A: pop an OS image window -------------
        Image.open(img_path).resize((224, 224)).show()

        # --- option B: inline Matplotlib (remove A if you pick B) ---
        # plt.imshow(Image.open(img_path), cmap="gray")
        # plt.axis("off")
        # plt.show()