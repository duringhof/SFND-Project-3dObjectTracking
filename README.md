# SFND 3D Object Tracking - Final project

This repository contains the submission for the final "Camera class" project related to "Tracking an object in 3D space", which is part of Udacity's Sensor Fusion Nanodegree program.

To obtain the starter code, read about dependencies and/or basic build instructions, please refer to the following repository:
https://github.com/udacity/SFND_3D_Object_Tracking.git

## Report

---

#### Task FP.0 Final Report
> Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.

This README is created to meet this specification.

---

#### Task FP.1 Match 3D Objects
> Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.

```c++
void matchBoundingBoxes(std::vector<cv::DMatch> &matches,
                        std::map<int, int> &bbBestMatches, DataFrame &prevFrame,
                        DataFrame &currFrame) {

  int p = prevFrame.boundingBoxes.size();
  int c = currFrame.boundingBoxes.size();
  int countMatches[p][c] = {};

  // loop across all keypoint matches
  for (cv::DMatch match : matches) {

    bool prevBB_found = false;
    bool currBB_found = false;
    std::vector<int> prevBB_idx, currBB_idx;

    // find bounding boxes in previous frame that contain the matched keypoint
    for (int i = 0; i < p; i++) {

      if (prevFrame.boundingBoxes[i].roi.contains(prevFrame.keypoints[match.queryIdx].pt)) {

        prevBB_found = true;
        prevBB_idx.push_back(i);
      }
    }

    // find bounding boxes in current frame that contain the matched keypoint
    for (int i = 0; i < c; i++) {

      if (currFrame.boundingBoxes[i].roi.contains(currFrame.keypoints[match.trainIdx].pt)) {

        currBB_found = true;
        currBB_idx.push_back(i);
      }
    }

    // increment counters for all possible bounding box mathches that the
    // current matched keypoint could imply
    if (prevBB_found && currBB_found) {

      for (auto itr_c : currBB_idx) {

        for (auto itr_p : prevBB_idx) {
          
          countMatches[itr_p][itr_c] += 1;
        }
      }
    }

    // for each bounding box in the previous frame, find the current frame
    // bounding box that matches according to the most keypoint matches
    for (int i = 0; i < p; i++) {

      int max_count = 0;
      int max_idc = 0;
      for (int j = 0; j < c; j++) {

        if (countMatches[i][j] > max_count) {
          
          max_count = countMatches[i][j];
          max_idc = j;
        }
      }
      bbBestMatches[i] = max_idc;
    }
  }
}
```

---

#### Task FP.2 Compute Lidar-based TTC
> Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.

```c++
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate,
                     double &TTC) {

  auto compareLidarPoints = [](LidarPoint lp1, LidarPoint lp2) {
    return (lp1.x < lp2.x);
  };

  double dT = 1 / frameRate;
  std::sort(lidarPointsPrev.begin(), lidarPointsPrev.end(), compareLidarPoints);
  std::sort(lidarPointsCurr.begin(), lidarPointsCurr.end(), compareLidarPoints);

  // select 10% of lidar points with closest distance (at least 1)
  int nPrev = lidarPointsPrev.size() < 10 ? 1 : lidarPointsPrev.size() / 10;
  int nCurr = lidarPointsCurr.size() < 10 ? 1 : lidarPointsCurr.size() / 10;

  // find median distance of selected lidar points
  double d_prev = (lidarPointsPrev[std::ceil(nPrev / 2. - 1)].x +
                   lidarPointsPrev[std::floor(nPrev / 2.)].x) /
                  2.0;
  double d_curr = (lidarPointsCurr[std::ceil(nCurr / 2. - 1)].x +
                   lidarPointsCurr[std::floor(nCurr / 2.)].x) /
                  2.0;

  TTC = (d_curr * dT) / (d_prev - d_curr);
}
```
#### Task FP.3 Associate Keypoint Correspondences with Bounding Boxes
> Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.

```c++
void clusterKptMatchesWithROI(BoundingBox &boundingBox,
                              std::vector<cv::KeyPoint> &kptsPrev,
                              std::vector<cv::KeyPoint> &kptsCurr,
                              std::vector<cv::DMatch> &kptMatches) {

  // find matches belonging to boundingbox and calculate euclidian distances
  vector<double> euclDistances;
  for (cv::DMatch match : kptMatches) {

    if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt)) {

      boundingBox.keypoints.push_back(kptsCurr[match.trainIdx]);
      boundingBox.kptMatches.push_back(match);
      double euclDist =
          cv::norm(kptsPrev[match.queryIdx].pt - kptsCurr[match.trainIdx].pt);
      euclDistances.push_back(euclDist);
    }
  }

  // calculate median euclidian distance
  int n = euclDistances.size();
  std::sort(euclDistances.begin(), euclDistances.end());
  double medianEuclDist = (euclDistances[std::ceil(n / 2. - 1)] +
                           euclDistances[std::floor(n / 2.)]) /
                          2.0;

  // filter out matches with sufficiently small deviation to the median euclidian distance
  vector<cv::KeyPoint> kpts;
  vector<cv::DMatch> matches;
  for (cv::DMatch match : boundingBox.kptMatches) {

    double euclDist =
        cv::norm(kptsPrev[match.queryIdx].pt - kptsCurr[match.trainIdx].pt);
    if (abs(euclDist - medianEuclDist) <= 25) {
      matches.push_back(match);
      kpts.push_back(kptsCurr[match.trainIdx]);
    }
  }
  boundingBox.keypoints = kpts;
  boundingBox.kptMatches = matches;  
}
```

---
#### Task FP.4 Compute Camera-based TTC
> Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.

```c++
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev,
                      std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate,
                      double &TTC, cv::Mat *visImg) {

  vector<double> distRatios;

  for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1) {

    // get current keypoint and its matched partner in the prev. frame
    cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
    cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

    for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) {

      double minDist = 100.0; // min. required distance

      // get next keypoint and its matched partner in the prev. frame
      cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
      cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

      // compute distances and distance ratios
      double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
      double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

      if (distPrev > std::numeric_limits<double>::epsilon() &&
          distCurr >= minDist) {

        double distRatio = distCurr / distPrev;
        distRatios.push_back(distRatio);
      }
    }
  }

  // only continue if list of distance ratios is not empty
  if (distRatios.size() == 0) {

    TTC = NAN;
    return;
  }

  // compute camera-based TTC from distance ratios
  double dT = 1 / frameRate;
  std::sort(distRatios.begin(), distRatios.end());
  double medianDistRatio = (distRatios[std::ceil(distRatios.size() / 2. - 1)] +
                            distRatios[std::floor(distRatios.size() / 2.)]) /
                           2.0;
  TTC = -dT / (1 - medianDistRatio);
}
```

---
#### Task FP.5 Performance Evaluation 1
> Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.

| Picture | Explanation |
| ---- | ---- |
| ![Lidar High Value](images/1.png) |  The picture shows which lidar points are consdered for distance measurement, red points are from prev frame and green from the current. In this example the lidar points considered switched from the rear bumper to the hatch|
![Lidar Low Value](images/2.png) |  Similar situation as previous, the considered lidar measurements have changed "focus" on the vehicle considerably|
![Lidar Very High Value](images/3.png) |  The measurements at the license plate appear to be more noisy|

---
#### Task FP.6 Performance Evaluation 2
> Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

```

```

---