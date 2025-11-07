import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

y = np.linspace(0, 2*np.pi, 100)
f = np.sin(y)
fig, ax = plt.subplots()
ax.plot(y, f)
buf = io.BytesIO()
fig.savefig(buf, format='png')
plt.close(fig)

img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
print("Base64 length:", len(img_str))

with open("test.png", "wb") as f:
    f.write(buf.getvalue())
