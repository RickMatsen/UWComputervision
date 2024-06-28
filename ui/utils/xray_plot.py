from matplotlib import pyplot as plt

import matplotlib.patches as patches
from PIL import Image
import io

ordered_labels = ['Humeral head fit',
 'Spinoglenoid notch',
 'Anterior lip (glenoid fossa)',
 'Posterior lip (glenoid fossa)',
 'Center (glenoid fossa)',
 'Scapular body']

def draw_pred(X, y, idx, num_ell=2, num_pts=4) -> None:
    
    colors = iter(['b','g','r','c','m','y'])
    labels = iter(ordered_labels)

    # the first 5 used to be ellipse
    ellipses = y[idx, :][:5]
    points = y[idx, :][5:]
    print(ellipses)
    # torch auto casts data to floating point numbers
    # to make the behavior of plt.imshow consistent
    # recast the image to int type

    plt.imshow(X[idx, :, :, :].permute(2, 1, 0).int())
    for i in range(num_ell):
        if i == 0: # 'Circle fit to humeral articular surface'
            cx, cy, rx= ellipses[0:3]
            ry = rx
            dx, dy = rx * 2, ry * 2
            plt.gca().add_patch(
            patches.Ellipse(xy=(cx, cy), width=dx, height=dy,fill=False,color=next(colors),label = next(labels)))

        if i == 1: # 'Spinoglenoid notch'
            x, y = ellipses[3:5]
            plt.scatter(x,y,color=next(colors),label = next(labels), s=60)
    

    for k in range(num_pts):
        x, y = points[2*k:2*(k+1)]
        plt.scatter(x,y,color=next(colors),label = next(labels), s=60)

    plt.legend()


def draw_pred_pil(X, y, idx, num_ell=2, num_pts=4):
    colors = iter(['b', 'g', 'r', 'c', 'm', 'y'])
    labels = iter(ordered_labels)

    # Assuming y is a numpy array or a tensor that can be indexed in this way
    ellipses = y[idx, :][:5]
    points = y[idx, :][5:]

    # Create a figure and an axes to draw on
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(X[idx, :, :, :].permute(2, 1, 0).int()) # Adjusted the permute for correct orientation

    for i in range(num_ell):
        if i == 0:  # 'Circle fit to humeral articular surface'
            cx, cy, rx = ellipses[0:3]
            ry = rx
            dx, dy = rx * 2, ry * 2
            ax.add_patch(patches.Ellipse(xy=(cx, cy), width=dx, height=dy, fill=False, color=next(colors), label=next(labels)))

        if i == 1:  # 'Spinoglenoid notch'
            x, y = ellipses[3:5]
            ax.scatter(x, y, color=next(colors), label=next(labels), s=60)

    for k in range(num_pts):
        x, y = points[2 * k:2 * (k + 1)]
        ax.scatter(x, y, color=next(colors), label=next(labels), s=60)

    ax.legend(bbox_to_anchor=(1, 0.5))
    ax.set_axis_off()

    # Save the figure to a bytes buffer and then to a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)

    # Close the figure to prevent memory leak
    plt.close(fig)

    return img