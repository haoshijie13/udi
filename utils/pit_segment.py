import os

import cv2
import numpy as np
import tifffile as tifi


def check_shape(ct):
    farthest = 0
    max_dist = 0
    for i in range(ct.shape[0]):
        d = np.sqrt((ct[i][0][0] - ct[i - 1][0][0]) ** 2 + (ct[i][0][1] - ct[i - 1][0][1]) ** 2)
        if d > max_dist:
            max_dist = d
            farthest = i

    rect = cv2.minAreaRect(ct)
    if rect[1][0] * rect[1][1] == 0:
        return True

    # if max_dist >= 25 and (abs(ct[farthest][0][0] - ct[farthest - 1][0][0]) < 1 or abs(ct[farthest][0][1] - ct[farthest - 1][0][1]) < 1):
    # if max_dist >= 25:
        # if rect[1][0] * rect[1][1] <= 2000 and (rect[1][0] / rect[1][1] >= 4 or rect[1][0] * rect[1][1] <= 0.25):
    if rect[1][0] / rect[1][1] >= 3 or rect[1][0] * rect[1][1] <= 1 / 3 or max_dist ** 2 > rect[1][0] * rect[1][1]:
    #     ratio = rect[1][0] / rect[1][1]
    #     if ratio < 0.5 or ratio > 2:
        return True

    return False


def match_pitpoints(pts):
    d_map = np.ones((len(pts), len(pts))) * np.inf
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            angle = abs(pts[i][1] - pts[j][1])
            if not (90 <= angle <= 270):
                continue
            if angle > 180:
                angle = 360 - angle

            # if abs(pts[i][0][0] - pts[j][0][0]) < 1 or abs(pts[i][0][1] - pts[j][0][1]) < 1:
            #     continue

            dist = np.sqrt((pts[i][0][0] - pts[j][0][0]) ** 2 + (pts[i][0][1] - pts[j][0][1]) ** 2)
            # dist *= (180 - angle) / angle + 1

            d_map[i][j] = dist

    matched = []
    while d_map.min() < np.inf:
        i, j = np.unravel_index(d_map.argmin(), d_map.shape)
        matched.append([pts[i][0], pts[j][0]])
        d_map[i, :] = np.inf
        d_map[j, :] = np.inf
        d_map[:, i] = np.inf
        d_map[:, j] = np.inf

    return matched


def get_angle(pt1, pt2):
    pt1_x, pt1_y = pt1
    pt2_x, pt2_y = pt2

    if pt1_x == pt2_x:
        angle = np.sign(pt2_y - pt1_y) * 90
    else:
        k = (pt2_y - pt1_y) / (pt1_x - pt2_x)
        angle = -1 * np.rad2deg(np.arctan(k))

        if pt1_x < pt2_x:
            angle = 180 - angle
        elif pt1_y > pt2_y:
            angle = 360 - angle

    return angle


def get_foot(pt, st, ed):
    pt_x, pt_y = pt
    st_x, st_y = st
    ed_x, ed_y = ed

    if st_x == ed_x:
        ft_x = st_x
        ft_y = pt_y
    else:
        k = (ed_y - st_y) / (ed_x - st_x)
        coef = st_y - k * st_x
        ft_x = round((pt_x + k * pt_y - k * coef) / (k ** 2 + 1))
        ft_y = round((k ** 2 * pt_y + k * pt_x + coef) / (k ** 2 + 1))

    return ft_x, ft_y


def split_child_pts(img, children, parent):
    for contour in children:
        cv2.drawContours(img, [contour], -1, 255, -1)
        # if len(contour) <= 2:
        #     continue
        # if cv2.contourArea(contour) < 60:
        #     cv2.drawContours(img, [contour], -1, 255, -1)
        #     continue
        # max_dist = 0
        # cp1 = []
        # cp2 = []
        # for i in range(len(contour)):
        #     for j in range(len(contour[i:])):
        #         p1 = contour[i][0]
        #         p2 = contour[j][0]
        #         if tuple(p1) == tuple(p2):
        #             continue
        #         dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        #         if dist > max_dist:
        #             max_dist = dist
        #             cp1 = p1
        #             cp2 = p2
        # min1 = np.inf
        # min2 = np.inf
        # pp1 = []
        # pp2 = []
        # for pp in parent:
        #     dist1 = np.sqrt((pp[0][0] - cp1[0]) ** 2 + (pp[0][1] - cp1[1]) ** 2)
        #     dist2 = np.sqrt((pp[0][0] - cp2[0]) ** 2 + (pp[0][1] - cp2[1]) ** 2)
        #     if dist1 < min1:
        #         min1 = dist1
        #         pp1 = pp[0]
        #     if dist2 < min2:
        #         min2 = dist2
        #         pp2 = pp[0]
        #
        # if len(cp1) * len(cp2) * len(pp1) * len(pp2) == 0:
        #     break
        #
        # cp1 = tuple(cp1)
        # cp2 = tuple(cp2)
        # pp1 = tuple(pp1)
        # pp2 = tuple(pp2)
        #
        # angle = abs(get_angle(pp1, cp1) - get_angle(pp2, cp2))
        # if not (90 <= angle <= 270):
        #     cv2.drawContours(img, [contour], -1, 255, -1)
        #     continue
        #
        # cv2.line(img, cp1, pp1, 0, 2)
        # cv2.line(img, cp2, pp2, 0, 2)

    return img


def get_dist(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def check_ft(ft, contour):
    min_dist = np.inf
    for i in range(len(contour)):
        pt = contour[i][0]
        dist = get_dist(ft, pt)
        if dist < min_dist:
            min_dist = dist
    if min_dist < 2:
        return False
    else:
        return True

def pit_detect(img):
    # contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchies = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    ret = img.copy()
    # ret[ret > 0] = 255

    for i in range(len(contours)):
        contour = contours[i]
        hierarchy = hierarchies[0][i]

        if hierarchy[3] != -1:
            continue
        else:
            if cv2.contourArea(contour) > 50000:
                ret = cv2.drawContours(ret, [contour], -1, 0, -1)
                continue

            if check_shape(contour): # and cv2.contourArea(contour) < 50:
                ret = cv2.drawContours(ret, [contour], -1, 0, -1)
                continue

            if hierarchy[2] != -1:
                children = [contours[j] for j in range(len(contours)) if hierarchies[0][j][3] == i]
                ret = split_child_pts(ret, children, contour)

    tmp = ret.copy()
    # tmp[tmp > 0] = 255
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    contours, _ = cv2.findContours(ret, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        pit_pts = []

        hull = cv2.convexHull(contour, True, returnPoints=False)
        tt = sorted([int(val) for val in hull])
        for i in range(len(hull)):
            hull[i] = np.array(tt[i])

        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            continue
        elif len(defects) == 0:
            continue
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            st = tuple(contour[s][0])
            ed = tuple(contour[e][0])
            fr = tuple(contour[f][0])
            cv2.line(tmp, st, ed, (0, 255, 255), 1)
            if d / 256 <= 10:
                continue
            cv2.circle(tmp, fr, 1, (255, 0, 0), -1)
            ft = get_foot(fr, st, ed)
            if not check_ft(ft, contour):
                continue
            angle = get_angle(ft, fr)
            cv2.line(tmp, fr, ft, (255, 255, 0), 1)

            pit_pts.append([fr, angle, i])

        if len(pit_pts) < 2:
            # hull = cv2.convexHull(contour)
            # cv2.drawContours(tmp, [hull], -1, (255, 255, 255), thickness=-1)
            # cv2.drawContours(ret, [hull], -1, 255, 1)
            for pt in pit_pts:
                idx = pt[2]
                dd = defects[idx, 0, 3]
                min_dist = np.inf
                mhp = []
                hh = cv2.convexHull(contour)
                for hp in hh:
                    dist = np.sqrt((hp[0][0] - pt[0][0]) ** 2 + (hp[0][1] - pt[0][1]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        mhp = hp

                if min_dist - dd < 0:
                    ang = abs(pt[1] - get_angle(mhp[0], pt[0]))
                    if 90 <= ang <= 270:
                        cv2.line(tmp, pt[0], mhp[0], (0, 0, 0), 2)
                        cv2.line(ret, pt[0], mhp[0], 0, 2)
            continue
        else:
            matched = match_pitpoints(pit_pts)
            for pair in matched:
                cv2.line(tmp, pair[0], pair[1], (0, 0, 0), 2)
                cv2.line(ret, pair[0], pair[1], 0, 2)

    if ret.sum() == img.sum():
        return ret, tmp, 0
    else:
        return ret, tmp, 1


def entry(img):
    flag = 1
    while flag > 0:
        img, tmp, flag = pit_detect(img)

    return img


if __name__ == '__main__':
    img_file = r'D:\test_data\singlecell\RNA_Segment\tst\SS200000576BL_A2\ttttt.tif'
    img = tifi.imread(img_file)
    entry(img)

