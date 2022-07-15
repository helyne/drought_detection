import plotly.express as px
import matplotlib.pyplot as plt
from drought_detection.data_handling import get_img_from_example


# plotyly style
# function to plot satellite images (incomplete (works for 1 image), also needs documentation)
def plot_sat_imgs(parsed_examples, channels=':', n_imgs=1):

    for i in range(n_imgs):
        img, label = img, label = get_img_from_example(parsed_examples[i])

        if channels == ':':
            fig = px.imshow(img[:,:,:], title=str(label))
            fig.show()
        else:
            channels = channels.strip('][').split(', ')
            s = slice(int(channels[0][0]), int(channels[0][-1]), 1)
            fig = px.imshow(img[:,:,s], title=str(label))
            fig.show()




# matplotlib style
# function to plot satellite images (incomplete (works for 1 image, all channels), also needs documentation)
def plot_sat_imgs_plt(parsed_examples, channels=':', n_imgs=1):

    figgy=plt.figure(figsize=(20, 30), dpi= 80, facecolor='w', edgecolor='k')

    for i in range(n_imgs):

        plt.subplot(5, 5, i+1)
        img, label = img, label = get_img_from_example(parsed_examples[i])

        if channels == ':':
            plt.imshow(img).axes.get_xaxis().set_visible(False)
            plt.imshow(img).axes.get_yaxis().set_visible(False);

        else:
            channels = channels.strip('][').split(', ')
            s = slice(int(channels[0][0]), int(channels[0][-1]))
            plt.imshow(img[:,:,s]).axes.get_xaxis().set_visible(False)
            plt.imshow(img[:,:,s]).axes.get_yaxis().set_visible(False)

        plt.title(str(label));
