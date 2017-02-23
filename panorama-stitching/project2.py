import sys
import math
import numpy as np
import cv2
import operator
import multiprocessing as mp
from random import shuffle
from scipy.spatial import cKDTree

class FeaturePoint(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.l = 0
        self.cos = 0
        self.sin = 0
        self.descriptor = None

class TranslationModel(object):
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.dx = 0
        self.dy = 0
        self.inlier_num = 0
        self.matching_pairs = []


class MatchingPair(object):
    def __init__(self, fp1, fp2, is_in=False):
        self.fp1 = fp1
        self.fp2 = fp2
        self.is_in = is_in

class ImageInfo(object):
    def __init__(self):
        self.pano_id = -1
        self.x = 0
        self.y = 0

THRESHOLD = 0.5
RANSAC_SAMPLE_NUM = 6
POTENTIAL_IMAGE_MATCHES = 6
RANSAC_INLIER_PROBABILITY = 0.6
RANSAC_INLIER_THRESHOLD = 9.0
RANSAC_SUCCESS_PROBABILITY = 0.99


"""""""""""""""""""""""""""""""""
        FEATURE DETECTION
"""""""""""""""""""""""""""""""""
def build_pyramid(img, level):
    gp = [None] * (level + 1)
    gp[0] = img

    for i in range(0, level):
        gp[i+1] = cv2.GaussianBlur(gp[i][::2, ::2], None, 1)

    return gp


def find_local_maxima(matrix):
    """
    non-local-maxima supression within 3x3 windows
    """
    t = 200.0
    local_maxima = cv2.dilate(matrix, np.ones((3, 3)))
    return matrix * ((matrix == local_maxima) & (matrix > t))


def adaptive_non_maximal_suppression(ip):
    n_ip = 500
    N = np.count_nonzero(ip)
    ip_loc = ip.nonzero()

    if N <= n_ip:
        return np.transpose(ip_loc)

    ip_f = ip[ip_loc]
    ip_loc = np.transpose(ip_loc)

    # sort coordinates by corresponding f in descending order
    sort_map = np.argsort(-ip_f)
    ip_loc = ip_loc[sort_map]

    dist = np.sqrt(np.sum(np.square(ip_loc[:, np.newaxis, :] - ip_loc), axis=2))

    height, width = ip.shape
    r = math.floor(
        ((height+width) + math.sqrt((height+width)**2 + 1996*height*width)) / 998)
    n = 0
    suppressed = None
    while True:
        suppressed = np.zeros_like(ip_f, dtype=bool)
        for i in range(1, N):
            suppressed[i:] |= np.less(dist[i-1, i:], r)

        n = np.count_nonzero(-suppressed)
        if (n >= n_ip):
            break;
        r -= 1

    ip_loc = ip_loc[(-suppressed).nonzero()[0][:n_ip]]

    return ip_loc


def harris_corner(img):
    src = cv2.GaussianBlur(img.astype(float), None, 1)
    Ix = cv2.Scharr(src, cv2.CV_64F, 1, 0)
    Iy = cv2.Scharr(src, cv2.CV_64F, 0, 1)
    Ixx = np.square(Ix)
    Iyy = np.square(Iy)
    Ixy = Ix * Iy
    Sxx = cv2.GaussianBlur(Ixx, None, 1.5)
    Syy = cv2.GaussianBlur(Iyy, None, 1.5)
    Sxy = cv2.GaussianBlur(Ixy, None, 1.5)

    R = (Sxx * Syy - Sxy * Sxy) / (Sxx + Syy + np.spacing(1))
    ip = find_local_maxima(R)

    # remove boundary points
    boundary = 60
    ip[:boundary, :] = 0
    ip[-boundary:, :] = 0
    ip[:, :boundary] = 0
    ip[:, -boundary:] = 0

    ip_loc = adaptive_non_maximal_suppression(ip)
    feature_points = [None] * ip_loc.shape[0]
    for i in range(0, len(feature_points)):
        feature_points[i] = FeaturePoint(x=ip_loc[i][1], y=ip_loc[i][0])

    return feature_points


def compute_orientation(img, feature_points):
    src = cv2.GaussianBlur(img.astype(float), None, 4.5)
    Ix = cv2.Scharr(src, cv2.CV_64F, 1, 0)
    Iy = cv2.Scharr(src, cv2.CV_64F, 0, 1)
    Ixx = np.square(Ix)
    Iyy = np.square(Iy)

    G = np.sqrt(Ixx + Iyy + np.spacing(1))
    cos = Ix / G
    sin = Iy / G

    for fp in feature_points:
        fp.cos = cos[fp.y, fp.x]
        fp.sin = sin[fp.y, fp.x]

    return feature_points


def extract_feature_descriptor(pyramid, feature_points, level):
    kernel = np.ones((5, 5), np.float32) / 25
    sample_img = cv2.filter2D(pyramid[level+1], -1, kernel)

    for fp in feature_points:
        rot_mat = np.matrix([[fp.cos, -fp.sin], [fp.sin, fp.cos]])
        offset = np.arange(-18, 20, 5)
        x = np.tile(offset, 8).ravel()
        y = offset.repeat(8)
        coord = np.int0(np.rint(rot_mat * np.array([x, y]))) + [[fp.x>>1], [fp.y>>1]]

        fp.descriptor = sample_img[coord[1], coord[0]][0]
        fp.descriptor = (fp.descriptor - fp.descriptor.mean()) / fp.descriptor.std()
        fp.x <<= level
        fp.y <<= level
        fp.l = level

    return feature_points


def extract_MSOP(pyramid):
    feature_points = []
    for l in range(0, len(pyramid)-1):
        fps = harris_corner(pyramid[l])
        fps = compute_orientation(pyramid[l], fps)
        feature_points += extract_feature_descriptor(pyramid, fps, l)

    return feature_points


def feature_detection(src_images):
    level = 3
    gaussian_pyramids = []
    for img in src_images:
        gaussian_pyramids.append(build_pyramid(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), level))

    img_features = []
    for pyr in gaussian_pyramids:
        img_features.append(extract_MSOP(pyr))

    return img_features


"""""""""""""""""""""""""""""""""
        FEATURE MATCHING
"""""""""""""""""""""""""""""""""
def feature_matching_task(img_features, index):
    img_num = len(img_features)

    # Construct KDtrees per image
    kdtrees = []
    for fp_list in img_features:
        kdtrees.append(cKDTree([fp.descriptor for fp  in fp_list]))

    k = 2 # KNN
    fp_list = []
    for fp in img_features[index]:
        best_match_list = np.zeros(img_num, dtype=np.int64)
        one_nn_dists = np.zeros(img_num)
        avg_2nn_dist = 0.0

        for j in range(0, img_num):
            if j == index:
                best_match_list[j] = -1
                continue
            dists, neighbors_index = kdtrees[j].query(fp.descriptor, k)
            best_match_list[j] = neighbors_index[0]
            one_nn_dists[j] = dists[0]
            avg_2nn_dist += dists[1]

        avg_2nn_dist /= (img_num - 1)

        for j in range(0, img_num):
            if one_nn_dists[j] > THRESHOLD * avg_2nn_dist:
                best_match_list[j] = -1

        fp_list.append(best_match_list)

    return fp_list


def feature_matching(img_features):
    process_list = [None] * len(img_features)
    pool = mp.Pool(processes=mp.cpu_count())
    for j in range(0, len(process_list)):
        process_list[j] = pool.apply_async(
            feature_matching_task, (img_features, j))
    pool.close()
    pool.join()

    matched_feature_list = []
    for p in process_list:
        matched_feature_list.append(p.get())

    for i in range(0, len(img_features)):
        for n in range(0, len(img_features[i])):
            for j in range(0, len(img_features)):
                if i == j:
                    continue
                elif matched_feature_list[i][n][j] == -1:
                    continue
                elif matched_feature_list[j][matched_feature_list[i][n][j]][i] != n:
                    matched_feature_list[i][n][j] = -1

    return matched_feature_list


"""""""""""""""""""""""""""""""""
        IMAGE MATCHING
"""""""""""""""""""""""""""""""""
def ransac(tm, img_features):
    K = math.trunc(math.log(1 - RANSAC_SUCCESS_PROBABILITY) /
        math.log(1.0 - math.pow(RANSAC_INLIER_PROBABILITY,
            RANSAC_SAMPLE_NUM)))
    src = tm.src
    dst = tm.dst

    for k in range(0, K):
        dx = 0
        dy = 0

        shuffle(tm.matching_pairs)
        i = 0
        while i < RANSAC_SAMPLE_NUM:
            fp1 = img_features[src][tm.matching_pairs[i].fp1]
            fp2 = img_features[dst][tm.matching_pairs[i].fp2]
            dx += fp2.x - fp1.x
            dy += fp2.y - fp1.y
            tm.matching_pairs[i].is_in = True
            i += 1

        inlier_num = RANSAC_SAMPLE_NUM
        dx /= float(RANSAC_SAMPLE_NUM)
        dy /= float(RANSAC_SAMPLE_NUM)

        for j in range(i, len(tm.matching_pairs)):
            fp1 = img_features[src][tm.matching_pairs[j].fp1]
            fp2 = img_features[dst][tm.matching_pairs[j].fp2]
            dist = (math.pow(fp2.x - fp1.x - dx, 2) +
                math.pow(fp2.y - fp1.y - dy, 2))
            if dist < RANSAC_INLIER_THRESHOLD:
                inlier_num += 1
                tm.matching_pairs[j].is_in = True

        if inlier_num > tm.inlier_num:
            tm.inlier_num = inlier_num
            tm.dx, tm.dy = dx, dy
        else:
            for p in tm.matching_pairs:
                p.is_in = False

    return tm


def image_matching(matched_feature_list, img_features):
    image_matching_list = []
    img_num = len(matched_feature_list)

    for i in range(0, img_num):
        tm_list = []
        feature_num = len(matched_feature_list[i])

        for j in range(0, img_num):
            if i == j:
                continue

            tm = TranslationModel(src=i, dst=j)

            for n in range(0, feature_num):
                if matched_feature_list[i][n][j] > -1:
                    tm.matching_pairs.append(MatchingPair(fp1=n,
                        fp2=matched_feature_list[i][n][j]))

            if len(tm.matching_pairs) > RANSAC_SAMPLE_NUM:
                tm_list.append(tm)

        tm_list.sort(key=lambda s: len(s.matching_pairs), reverse=True)
        if len(tm_list) > POTENTIAL_IMAGE_MATCHES:
            tm_list = tm_list[:POTENTIAL_IMAGE_MATCHES]

        j = 0
        while j < len(tm_list):
            tm_list[j] = ransac(tm_list[j], img_features)
            if (tm_list[j].inlier_num - 5.9 <= 0.22 * len(tm_list[j].matching_pairs)):
                del tm_list[j]
                j -= 1

            j += 1

        image_matching_list.append(tm_list)

    return image_matching_list


"""""""""""""""""""""""""""""""""
        IMAGE STITCHING
"""""""""""""""""""""""""""""""""
def blend_linear_interpolation(img1, img2, x, y, left=False):
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # find need-blending section
    start_x = 0.0
    end_x = width2 - 1.0

    if left:
        for j in range(0, width2):
            if np.count_nonzero(img1_gray[y:y+height2, j]) > 0:
                start_x = j
                break

        for j in range(width2-1, -1, -1):
            if np.count_nonzero(img2_gray[:height2, j]) > 0:
                end_x = j
                break
    else:
        for j in range(0, width2):
            if np.count_nonzero(img2_gray[:height2, j]) > 0:
                start_x = j
                break

        for j in range(width1-x-1, -1, -1):
            if np.count_nonzero(img1_gray[y:y+height2, x+j]) > 0:
                end_x = j
                break

    # interpolation
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    for j in range(0, width2):
        w = np.clip(float(j-start_x)/(end_x-start_x), 0.0, 1.0)
        w = w if left else 1.0 - w

        for i in range(0, height2):
            if img1_gray[y+i][x+j] == 0:
                img1[y+i][x+j] = img2[i][j]
            elif img2_gray[i][j] > 0:
                img1[y+i][x+j] = w * img1[y+i][x+j] + (1.0-w) * img2[i][j]

    return img1.astype(np.uint8)


def stitch_image(pano, imgs, img_matching_list, src, dst, img_infos):
    dst_src_index = -1
    for i in range(0, len(img_matching_list[src])):
        if img_matching_list[src][i].dst == dst:
            dst_src_index = i
            break

    src_dst_index = -1
    for i in range(0, len(img_matching_list[dst])):
        if img_matching_list[dst][i].dst == src:
            src_dst_index = i
            break

    pano_height, pano_width = pano.shape[:2]
    dst_height, dst_width = imgs[dst].shape[:2]

    dx = 0
    dy = 0
    if dst_src_index > -1:
        if (src_dst_index < 0 or img_matching_list[src][dst_src_index].inlier_num > img_matching_list[dst][src_dst_index].inlier_num):
            dx = int(round(img_matching_list[src][dst_src_index].dx))
            dy = int(round(img_matching_list[src][dst_src_index].dy))
        else:
            dx = -int(round(img_matching_list[dst][src_dst_index].dx))
            dy = -int(round(img_matching_list[dst][src_dst_index].dy))
    else:
        dx = -int(round(img_matching_list[dst][src_dst_index].dx))
        dy = -int(round(img_matching_list[dst][src_dst_index].dy))


    min_x = min(0, img_infos[src].x - dx)
    min_y = min(0, img_infos[src].y - dy)
    max_x = max(pano_width, img_infos[src].x + dst_width - dx)
    max_y = max(pano_height, img_infos[src].y + dst_height - dy)

    """
    the image will be stitched at the upper left of new panorama
    update all image's relative coordinates in current panorama
    """
    if min_x < 0 or min_y < 0:
        for i in img_infos:
            if i.pano_id == img_infos[src].pano_id:
                i.x -= min_x
                i.y -= min_y

    img_infos[dst].x = img_infos[src].x - dx
    img_infos[dst].y = img_infos[src].y - dy
    img_infos[dst].pano_id = img_infos[src].pano_id

    new_pano = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=np.uint8)
    new_pano[-min_y:pano_height-min_y, -min_x:pano_width-min_x] = pano
    new_pano = blend_linear_interpolation(new_pano, imgs[dst], img_infos[dst].x,
        img_infos[dst].y, dx > 0)

    for i in range(0, len(img_matching_list[dst])):
        if img_infos[img_matching_list[dst][i].dst].pano_id < 0:
            new_pano = stitch_image(new_pano, imgs, img_matching_list,
                dst, img_matching_list[dst][i].dst, img_infos)

    return new_pano


def image_stitching(imgs, img_matching_list):
    img_infos = [ImageInfo() for i in range(0, len(imgs))]

    panoramas = []
    src = 0
    while src > -1:
        img_infos[src].pano_id = len(panoramas)
        pano = imgs[src]

        for i in range(0, len(img_matching_list[src])):
            dst = img_matching_list[src][i].dst
            if img_infos[dst].pano_id > -1:
                pano = stitch_image(panoramas[img_infos[dst].pano_id], imgs,
                    img_matching_list, dst, src, img_infos)

        for i in range(0, len(img_matching_list[src])):
            dst = img_matching_list[src][i].dst
            if img_infos[dst].pano_id < 0:
                pano = stitch_image(pano, imgs,
                    img_matching_list, src, dst, img_infos)

        if img_infos[src].pano_id == len(panoramas):
            panoramas.append(pano)
        else:
            panoramas[img_infos[src].pano_id] = pano

        src = -1
        for i in range(0, len(img_infos)):
            if img_infos[i].pano_id < 0:
                src = i
                break

    img_infos.sort(key=operator.attrgetter('x'))
    pano_img_list = [[] for i in range(0, len(panoramas))]

    for i in range(0, len(img_infos)):
        pano_img_list[img_infos[i].pano_id].append(img_infos[i])

    for i in range(0, len(panoramas)):
        x0 = pano_img_list[i][0].x
        y0 = pano_img_list[i][0].y
        xn = pano_img_list[i][-1].x
        yn = pano_img_list[i][-1].y
        half_delta_y = int(abs(yn - y0) / 2)
        height, width = panoramas[i].shape[:2]
        angle = math.degrees(math.atan2(yn-y0, xn-x0))
        M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        panoramas[i] = cv2.warpAffine(panoramas[i], M, (width, height))
        panoramas[i] = panoramas[i][half_delta_y:-half_delta_y]

    # crop panoramas
    for i in range(0, len(panoramas)):
        height, width = panoramas[i].shape[:2]
        crop_h = int(0.05 * height)
        crop_w = int(0.015 * width)
        panoramas[i] = panoramas[i][crop_h:-crop_h, crop_w:-crop_w]

    return panoramas

"""""""""""""""""""""""""""""""""
     CYLINDRICAL PROJECTION
"""""""""""""""""""""""""""""""""
def bilinear_interpolate(img, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, img.shape[1]-1)
    x1 = np.clip(x1, 0, img.shape[1]-1)
    y0 = np.clip(y0, 0, img.shape[0]-1)
    y1 = np.clip(y1, 0, img.shape[0]-1)

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return (Ia*wa[:, np.newaxis] + Ib*wb[:, np.newaxis] + Ic*wc[:, np.newaxis] + Id*wd[:, np.newaxis])


def cylindrical_projection(img, focal_len):
    height, width, depth = img.shape
    center_x = (float)(width-1)/2
    center_y = (float)(height-1)/2

    x = np.arange(width) - center_x
    x = focal_len * np.tan(x/focal_len)
    r = np.sqrt(np.square(x) + math.pow(focal_len, 2)) / focal_len
    x += center_x
    x = np.tile(x, height)
    y = np.arange(height) - center_y
    y = np.outer(y, r) + center_y

    return (bilinear_interpolate(img, x, y.ravel()).reshape(height, width, depth).astype(np.uint8))


def draw_matching_pair(imgs, image_matching_list, img_features):
    for i in range(0, len(imgs)):
        for tm in image_matching_list[i]:
            height1, width1 = imgs[tm.src].shape[:2]
            height2, width2 = imgs[tm.dst].shape[:2]

            img = np.zeros((max(height1, height2), width1 + width2, 3),
                dtype=np.uint8)
            img[:height1, :width1] = imgs[tm.src]
            img[:height2, width1:] = imgs[tm.dst]

            for p in tm.matching_pairs:
                x1 = img_features[tm.src][p.fp1].x
                y1 = img_features[tm.src][p.fp1].y
                x2 = img_features[tm.dst][p.fp2].x + width1
                y2 = img_features[tm.dst][p.fp2].y
                cv2.circle(img, (x1, y1), 3, (0, 255, 0), -1)
                cv2.circle(img, (x2, y2), 3, (0, 255, 0), -1)
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

            cv2.imwrite('match/matched_%d_%d.jpg' % (tm.src, tm.dst), img)



def main(argv):
    info_list = argv[0]

    '''read image name and focal length from info.txt'''
    img_list = []
    with open(info_list, "r") as f:
        for eachline in f:
            img_list.append(eachline.split())

    '''do cylindrical projection on each image and store them in a list'''    
    src_images = []
    for i in img_list:
        print (i[0], i[1])
        src_images.append(cylindrical_projection(cv2.imread(i[0], 1), float(i[1])))
    num = 0
    for i in src_images:
        cv2.imwrite('cylin'+str(num)+'.jpg', i)
        num += 1

    '''feature dectection'''
    img_features = feature_detection(src_images)

    '''feature matching'''
    matched_feature_list = feature_matching(img_features)

    '''image matching'''
    img_matching_list = image_matching(matched_feature_list, img_features)

    '''draw matching points'''
    draw_matching_pair(src_images, img_matching_list, img_features)
    
    '''image stitching'''
    panoramas = image_stitching(src_images, img_matching_list)

    '''write output'''
    for i in range(0, len(panoramas)):
        cv2.imwrite('pano'+str(i)+'.jpg', panoramas[i])

if __name__ == '__main__':
    main(sys.argv[1:])
