(ns gp-blog.post-0
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.protocols :as mp]
            [vizard [core :refer :all] [plot :as plot]])
  (:import [java.util Random]))

;; need a core.matrix implementation that has cholesky decomposition
(m/set-current-implementation :vectorz)

;; generates n standard normals
(defn sample-gaussian [n]
  (let [rng (Random.)]
    (repeatedly n #(.nextGaussian rng))))

;; mean and cov can just be clojure data structures.
;; mean - vec or list, cov and a vec of vecs, etc
(defn sample-multivariate-gaussian [mean cov]
  (let [n (count mean)
        e (m/scale (m/identity-matrix n) 1e-8)
        L (:L (mp/cholesky (m/add (m/matrix cov) e)
                           {:results [:L]}))
        u (m/matrix (sample-gaussian n))
        samples (m/add (m/matrix mean)
                       (m/mmul L u))]
    (m/to-nested-vectors samples)))

;; the x values at which we want to sample values of f
(def test-xs (range -5 5 0.03))

;; kernel function - for generating infinitely differentiable functions
(defn squared-exponential [sigma2 lambda x y]
  (* sigma2 (Math/exp (* -0.5 (Math/pow (/ (- x y) lambda) 2)))))

;; applies an arity 2 function to all pairs of xs
;; and ys and returns a matrix of the results
(defn covariance-mat [f xs ys]
  (let [rows (count xs)
        cols (count ys)]
    (partition cols
               (for [i (range rows) j (range cols)]
                 (f (nth xs i) (nth ys j))))))

;; makes squared exponential covariance matrix
(defn sq-exp-cov [s2 l xs ys]
  (covariance-mat (partial squared-exponential s2 l) xs ys))

;; this is all we need to sample our GP at the test points
(def prior-mean (repeat (count test-xs) 0.0))
(def prior-cov (sq-exp-cov 1 1 test-xs test-xs))

;; convert xs and ys into something vizard can use.
;; need to names the points with col so multiple series can be plotted at once
(defn vizard-pts [xs ys col]
  (map (fn [x y] {:x x :y y :col col}) xs ys))

;; generates multiple function samples from the GP and converts to vizard points
(defn line-data [xs mean cov num-samples label]
  (flatten
   (conj
    (for [i (range num-samples)]
      (vizard-pts xs
                  (sample-multivariate-gaussian mean cov)
                  (str label " sample " i)))
    (vizard-pts xs mean (str label " mean")))))

;; generate data for plotting the confidence bands
(defn conf-data [xs mean cov]
  (let [std-dev (map #(Math/sqrt %) (m/diagonal cov))]
    (map (fn [x m s]
           {:x x :y (+ m (* 2 s)) :y2 (- m (* 2 s))})
         xs mean std-dev)))

(start-plot-server!)

;; plot/vizard plots the line data, but we need to poke
;; some additional data and marks information into the vega spec
;; to get the confidence bands on the plot
(defn plot-gp [line-data conf-data]
  (plot! (-> (plot/vizard {:mark-type :line
                           :color "category20b"}
                          line-data)
             (assoc-in [:data 1]
                       {:name :confidence
                        :values conf-data})
             (assoc-in [:marks 1]
                       {:type :area
                        :from {:data :confidence}
                        :properties
                        {:enter
                         {:x {:scale "x" :field :x}
                          :y {:scale "y" :field :y}
                          :y2 {:scale "y" :field :y2}
                          :interpolate {:value :monotone}
                          :fill {:value "#666"}}
                         :update {:fillOpacity {:value 0.25}}}}))))

;; sample from our GP for smooth functions and plot the results
(let [prior-data (line-data test-xs prior-mean prior-cov 3 "prior")
      prior-conf-data (conf-data test-xs prior-mean prior-cov)]
  (plot-gp prior-data prior-conf-data))

;; do the same with a kernel that doesn't generate smooth functions
(comment
  (defn abs-exponential [sigma2 lambda x y]
    (* sigma2 (Math/exp (* -0.5 (Math/abs (/ (- x y) lambda))))))

  (defn abs-exp-cov [s2 l xs ys]
    (pairwise-apply (partial abs-exponential s2 l) xs ys))

  (def abs-prior-cov (abs-exp-cov 1 1 test-xs test-xs))

  (let [prior-data (line-data test-xs prior-mean abs-prior-cov 3 "prior")
        prior-conf-data (conf-data test-xs prior-mean prior-cov)]
    (plot-gp prior-data prior-conf-data)))
