def draw_pred(X, y, idx, num_ell=2, num_pts=4) -> None:
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches
    ordered_labels = ['Circle fit to humeral articular surface',
 'Spinoglenoid notch',
 'Anterior lip of glenoid fossa',
 'Posterior lip of glenoid fossa',
 'Center of glenoid fossa',
 'Scapular body point']
    
    colors = iter(['b','g','r','c','m','y'])
    labels = iter(ordered_labels)

    # the first 5 used to be ellipse
    ellipses = y[idx, :][:5]
    points = y[idx, :][5:]
    print(ellipses)
    # torch auto casts data to floating point numbers
    # to make the behavior of plt.imshow consistent
    # recast the image to int type

    plt.imshow(X[idx, :, :, :].permute(1, 2, 0).int())
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

