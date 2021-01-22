import matplotlib.pyplot as plt
import SFM
import numpy as np


# def visualize_prediction(color, prediction):
#     xs_color = []
#     ys_color = []
#     for i in range(len(prediction[0])):
#         if (prediction[1][i] == color):
#             position = prediction[0][i]
#             xs_color.append(position[0])
#             ys_color.append(position[1])
#     plt.plot(xs_color, ys_color, 'ro', color=color[0], markersize=4)
# import matplotlib.pyplot as plt
#
class visualizer():
    def __init__(self,tfl,red_pos,green_pos,current_container,prediction):
        self.tfl = tfl
        self.image = tfl.img
        self.red_pos = red_pos
        self.green_pos = green_pos
        self.current_container = current_container
        self.prediction = prediction

    def visualize_detect_tfl(self):
        plt.plot(self.red_pos[0], self.red_pos[1], 'ro', color='r', markersize=4)
        plt.plot(self.green_pos[0], self.green_pos[1], 'ro', color='g', markersize=4)

    def visualize_prediction(self,color):
        xs_color = []
        ys_color = []
        for i in range(len(self.prediction[0])):
            if (self.prediction[1][i]== color):
                position = self.prediction[0][i]
                xs_color.append(position[0])
                ys_color.append(position[1])
        plt.plot(xs_color, ys_color, 'ro', color=color[0], markersize=4)

    def visualize_CNN (self):
        self.visualize_prediction("red")
        self.visualize_prediction("green")

    def run(self):
        if self.tfl.prev_container:
            plt.figure(figsize=(3, 1))
            plt.imshow(self.image)
            self.visualize_detect_tfl()
            self.visualize_CNN()
            self.visualize_SFM(self.tfl)
            plt.show()


# def viz_my_data(images, labels, predictions=None, num=(5, 5), labels2name={0: 'No TFL', 1: 'green TFL', 2: 'red TFL'}):
#     assert images.shape[0] == labels.shape[0]
#     assert predictions is None or predictions.shape[0] == images.shape[0]
#     h = 5
#     n = num[0] * num[1]
#     ax = plt.subplots(num[0], num[1], figsize=(h * num[0], h * num[1]), gridspec_kw={'wspace': 0.05}, squeeze=False,
#                       sharex=True, sharey=True)[1]  # .flatten()
#     idxs = np.random.randint(0, images.shape[0], n)
#     for i, idx in enumerate(idxs):
#         ax.flatten()[i].imshow(images[idx])
#         title = labels2name[labels[idx]]
#         if predictions is not None:
#             pre = predictions[idx][labels[idx]] * 100
#             pre = round(pre, 2)
#             # title += ' Prediction:NO {:.2f} ,G {:.2f} ,R {:.2f}'.format(predictions[idx][labels[0]],predictions[idx][labels[1]],predictions[idx][labels[0]])
#             title += ' Prediction: {0} %'.format(pre)
#
#         ax.flatten()[i].set_title(title)

    def visualize_SFM(self,tfl):
        norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = SFM.prepare_3D_data(tfl.prev_container, tfl.curr_container,
                                                                            tfl.focal, tfl.pp)
        norm_rot_pts = SFM.rotate(norm_prev_pts, R)
        rot_pts = SFM.unnormalize(norm_rot_pts, tfl.focal, tfl.pp)
        foe = np.squeeze(SFM.unnormalize(np.array([norm_foe]), tfl.focal, tfl.pp))

        fig, (curr_sec, prev_sec) = plt.subplots(1, 2, figsize=(12, 6))
        prev_sec.set_title('prev(' + tfl.prev_container.frame_id + ')')
        prev_sec.imshow(tfl.prev_container.img)
        prev_p = tfl.prev_container.traffic_light
        prev_sec.plot(prev_p[:, 0], prev_p[:, 1], 'b+')

        curr_sec.set_title('curr(' + tfl.curr_container.frame_id + ')')
        curr_sec.imshow(tfl.curr_container.img)
        curr_p = tfl.curr_container.traffic_light
        curr_sec.plot(curr_p[:, 0], curr_p[:, 1], 'b+')

        for i in range(len(curr_p)):
            curr_sec.plot([curr_p[i, 0], foe[0]], [curr_p[i, 1], foe[1]], 'b')
            if tfl.curr_container.valid[i]:
                curr_sec.text(curr_p[i, 0], curr_p[i, 1],
                              r'{0:.1f}'.format(tfl.curr_container.traffic_lights_3d_location[i, 2]), color='r')
        curr_sec.plot(foe[0], foe[1], 'r+')
        curr_sec.plot(rot_pts[:, 0], rot_pts[:, 1], 'g+')
        plt.show()



