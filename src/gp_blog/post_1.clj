(ns gp-blog.post-1
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.protocols :as mp]
            [gp-blog.post-0 :refer [test-xs line-data conf-data plot-gp sq-exp-cov]]
            [vizard [core :refer [start-plot-server!]]]))

(start-plot-server!)

(def train-xs [-4 -3 -1 0 1])
(def train-fs [-2 0 1 2 -1])

;; noiseless observations
(defn k-k-inv* [cov-fn test-xs train-xs]
  (let [k* (m/matrix (cov-fn test-xs train-xs))
        k (m/matrix (cov-fn train-xs train-xs))]
    (m/mmul k* (m/inverse k))))

(defn posterior-mean [cov-fn test-xs train-xs train-fs]
  (m/to-nested-vectors
   (m/mmul (k-k-inv* cov-fn test-xs train-xs)
           (m/matrix train-fs))))

(defn posterior-covariance [cov-fn test-xs train-xs]
  (m/to-nested-vectors
   (m/sub (m/matrix (cov-fn test-xs test-xs))
          (m/mmul (k-k-inv* cov-fn test-xs train-xs)
                  (m/matrix (cov-fn train-xs test-xs))))))

(let [cov-fn (partial sq-exp-cov 1 1)
      post-mean (posterior-mean cov-fn test-xs train-xs train-fs)
      post-cov (posterior-covariance cov-fn test-xs train-xs)
      post-data (line-data test-xs post-mean post-cov 3 "posterior")
      post-conf-data (conf-data test-xs post-mean post-cov)]
  (plot-gp post-data post-conf-data))


;; regression with additive gaussian noise
(defn k-k-inv* [cov-fn sigma2-bar test-xs train-xs]
  (let [n (count train-xs)
        k* (m/matrix (cov-fn test-xs train-xs))
        k (m/matrix (cov-fn train-xs train-xs))]
    (m/mmul k*
            (m/inverse
             (m/add k
                    (m/scale (m/identity-matrix n) sigma2-bar))))))

(defn posterior-mean
  [cov-fn sigma2-bar test-xs train-xs train-fs]
  (m/to-nested-vectors
   (m/mmul (k-k-inv* cov-fn sigma2-bar test-xs train-xs)
           (m/matrix train-fs))))

(defn posterior-covariance
  [cov-fn sigma2-bar test-xs train-xs]
  (m/to-nested-vectors
   (m/sub (m/matrix (cov-fn test-xs test-xs))
          (m/mmul (k-k-inv* cov-fn sigma2-bar test-xs train-xs)
                  (m/matrix (cov-fn train-xs test-xs))))))

(let [lambda 1.0
      sigma2 1.0
      sigma2-bar 0.05
      cov-fn (partial sq-exp-cov sigma2 lambda)
      post-mean (posterior-mean
                 cov-fn sigma2-bar test-xs train-xs train-fs)
      post-cov (posterior-covariance cov-fn sigma2-bar test-xs train-xs)
      post-data (line-data test-xs post-mean post-cov 3 "posterior")
      post-conf-data (conf-data test-xs post-mean post-cov)]
  (plot-gp post-data post-conf-data))


(let [lambda 0.1
      sigma2 4.0
      sigma2-bar 0.05
      cov-fn (partial sq-exp-cov sigma2 lambda)
      post-mean (posterior-mean
                 cov-fn sigma2-bar test-xs train-xs train-fs)
      post-cov (posterior-covariance cov-fn sigma2-bar test-xs train-xs)
      post-data (line-data
                 test-xs post-mean post-cov 3 "larger variance")
      post-conf-data (conf-data test-xs post-mean post-cov)]
  (plot-gp post-data post-conf-data))

(let [lambda 0.1
      sigma2 1.0
      sigma2-bar 0.05
      cov-fn (partial sq-exp-cov sigma2 lambda)
      post-mean (posterior-mean
                 cov-fn sigma2-bar test-xs train-xs train-fs)
      post-cov (posterior-covariance cov-fn sigma2-bar test-xs train-xs)
      post-data (line-data
                 test-xs post-mean post-cov 3 "smaller lambda")
      post-conf-data (conf-data test-xs post-mean post-cov)]
  (plot-gp post-data post-conf-data))

(let [lambda 2.0
      sigma2 1.0
      sigma2-bar 0.05
      cov-fn (partial sq-exp-cov sigma2 lambda)
      post-mean (posterior-mean
                 cov-fn sigma2-bar test-xs train-xs train-fs)
      post-cov (posterior-covariance cov-fn sigma2-bar test-xs train-xs)
      post-data (line-data test-xs post-mean post-cov 3 "larger lambda")
      post-conf-data (conf-data test-xs post-mean post-cov)]
  (plot-gp post-data post-conf-data))
