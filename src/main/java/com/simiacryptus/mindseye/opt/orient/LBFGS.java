/*
 * Copyright (c) 2017 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.opt.orient;

import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.eval.Trainable.PointSample;
import com.simiacryptus.mindseye.lang.Delta;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchPoint;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.util.ArrayUtil;

import java.util.*;
import java.util.stream.Collectors;

import static com.simiacryptus.util.ArrayUtil.add;
import static com.simiacryptus.util.ArrayUtil.dot;

/**
 * The type Lbfgs.
 * See also https://papers.nips.cc/paper/5333-large-scale-l-bfgs-using-mapreduce (Large-scale L-BFGS using MapReduce)
 */
public class LBFGS implements OrientationStrategy {
  
  /**
   * The History.
   */
  public final TreeSet<PointSample> history = new TreeSet<>(Comparator.comparing(x->-x.getMean()));
  private int minHistory = 3;
  private int maxHistory = 30;
  
  /**
   * The Verbose.
   */
  protected boolean verbose = false;
  
  @Override
  public LineSearchCursor orient(Trainable subject, PointSample measurement, TrainingMonitor monitor) {
    addToHistory(measurement, monitor);
    List<PointSample> history = Arrays.asList(this.history.toArray(new PointSample[]{}));
    DeltaSet result = lbfgs(measurement, monitor, history);
    SimpleLineSearchCursor returnValue;
    if(null == result) {
      returnValue = cursor(subject, measurement, "GD", measurement.delta.scale(-1));
    } else {
      returnValue = cursor(subject, measurement, "LBFGS", result);
    }
    while (this.history.size() > ((null==result)?minHistory:maxHistory)) {
      PointSample remove = this.history.pollFirst();
      if (verbose) {
        monitor.log(String.format("Removed measurement %s to history. Total: %s", Long.toHexString(System.identityHashCode(remove)), history.size()));
      }
    }
    return returnValue;
  }
  
  @Override
  public void reset() {
    history.clear();
  }
  
  /**
   * Add to history.
   *
   * @param measurement the measurement
   * @param monitor     the monitor
   */
  public void addToHistory(PointSample measurement, TrainingMonitor monitor) {
    PointSample copyFull = measurement.copyFull();
    if (!copyFull.delta.stream().flatMapToDouble(y -> Arrays.stream(y.getDelta())).allMatch(d -> Double.isFinite(d))) {
      if (verbose) monitor.log("Corrupt measurement");
    }
    else if (!copyFull.weights.stream().flatMapToDouble(y -> Arrays.stream(y.getDelta())).allMatch(d -> Double.isFinite(d))) {
      if (verbose) monitor.log("Corrupt measurement");
    }
    else if (history.isEmpty() || !history.stream().filter(x -> x.sum <= copyFull.sum).findAny().isPresent()) {
      if (verbose) {
        monitor.log(String.format("Adding measurement %s to history. Total: %s", Long.toHexString(System.identityHashCode(copyFull)), history.size()));
      }
      history.add(copyFull);
    }
  }
  
  protected DeltaSet lbfgs(PointSample measurement, TrainingMonitor monitor, List<PointSample> history) {
    DeltaSet result = measurement.delta.scale(-1);
    if (history.size() > minHistory) {
      DeltaSet original = result.copy();
      DeltaSet p = measurement.delta.copy();
      assert (p.stream().parallel().allMatch(y -> Arrays.stream(y.getDelta()).allMatch(d -> Double.isFinite(d))));
      double[] alphas = new double[history.size()];
      for (int i = history.size() - 2; i >= 0; i--) {
        DeltaSet sd = history.get(i + 1).weights.subtract(history.get(i).weights);
        DeltaSet yd = history.get(i + 1).delta.subtract(history.get(i).delta);
        double denominator = sd.dot(yd);
        if (0 == denominator) {
          monitor.log("Orientation vanished. Popping history element from " + history.stream().map(x->String.format("%s",x.getMean())).reduce((a,b)->a+", "+b));
          return lbfgs(measurement, monitor, history.subList(0, history.size() - 1));
        }
        alphas[i] = p.dot(sd) / denominator;
        p = p.subtract(yd.scale(alphas[i]));
        assert (p.stream().parallel().allMatch(y -> Arrays.stream(y.getDelta()).allMatch(d -> Double.isFinite(d))));
      }
      DeltaSet sk = history.get(history.size() - 1).weights.subtract(history.get(history.size() - 2).weights);
      DeltaSet yk = history.get(history.size() - 1).delta.subtract(history.get(history.size() - 2).delta);
      p = p.scale(sk.dot(yk) / yk.dot(yk));
      assert (p.stream().parallel().allMatch(y -> Arrays.stream(y.getDelta()).allMatch(d -> Double.isFinite(d))));
      for (int i = 0; i < history.size() - 1; i++) {
        DeltaSet sd = history.get(i + 1).weights.subtract(history.get(i).weights);
        DeltaSet yd = history.get(i + 1).delta.subtract(history.get(i).delta);
        double beta = p.dot(yd) / sd.dot(yd);
        p = p.add(sd.scale(alphas[i] - beta));
        assert (p.stream().parallel().allMatch(y -> Arrays.stream(y.getDelta()).allMatch(d -> Double.isFinite(d))));
      }
      for (Map.Entry<NNLayer, Delta> e : result.map.entrySet()) {
        double[] delta = p.map.get(e.getKey()).getDelta();
        Arrays.setAll(e.getValue().getDelta(), j -> delta[j]);
      }
      double mag = Math.sqrt(result.dot(result));
      double magGrad = Math.sqrt(original.dot(original));
      double dot = result.dot(original) / (mag * magGrad);
      Map<String, String> anglesPerLayer = measurement.delta.map.entrySet().stream()
                                             .collect(Collectors.toMap((Map.Entry<NNLayer, Delta> e) -> {
                                               return measurement.delta.map.get(e.getKey()).layer.getName();
                                             }, (Map.Entry<NNLayer, Delta> e) -> {
                                               double[] lbfgsVector = result.map.get(e.getKey()).getDelta();
                                               for(int index=0;index<lbfgsVector.length;index++) lbfgsVector[index] = Double.isFinite(lbfgsVector[index])?lbfgsVector[index]:0;
                                               double[] gradientVector = original.map.get(e.getKey()).getDelta();
                                               for(int index=0;index<gradientVector.length;index++) gradientVector[index] = Double.isFinite(gradientVector[index])?gradientVector[index]:0;
                                               double lbfgsMagnitude = ArrayUtil.magnitude(lbfgsVector);
                                               double gradientMagnitude = ArrayUtil.magnitude(gradientVector);
                                               assert Double.isFinite(gradientMagnitude);
                                               //assert Double.isFinite(lbfgsMagnitude);
                                               if (gradientMagnitude == 0.0) {
                                                 return String.format("%.3e", lbfgsMagnitude);
                                               }
                                               else {
                                                 double dotP = ArrayUtil.dot(lbfgsVector, gradientVector) / (lbfgsMagnitude * gradientMagnitude);
                                                 return String.format("%.3f/%.3e", dotP, lbfgsMagnitude / gradientMagnitude);
                                               }
                                             }));
      monitor.log(String.format("LBFGS Orientation magnitude: %.3e, gradient %.3e, dot %.3f; %s", mag, magGrad, dot, anglesPerLayer));
    }
    else {
      monitor.log(String.format("LBFGS Accumulation History: %s points", history.size()));
      return null;
    }
    if (!accept(measurement.delta, result)) {
      monitor.log("Orientation rejected. Popping history element from " + history.stream().map(x->String.format("%s",x.getMean())).reduce((a,b)->a+", "+b));
      return lbfgs(measurement, monitor, history.subList(0, history.size() - 1));
    } else {
      this.history.clear();
      this.history.addAll(history);
      return result;
    }
  }
  
  private SimpleLineSearchCursor cursor(Trainable subject, PointSample measurement, String type, DeltaSet result) {
    return new SimpleLineSearchCursor(subject, measurement, result) {
      public LineSearchPoint step(double t, TrainingMonitor monitor) {
        LineSearchPoint measure = super.step(t, monitor);
        addToHistory(measure.point, monitor);
        return measure;
      }
    }
             .setDirectionType(type);
  }
  
  protected boolean accept(DeltaSet gradient, DeltaSet direction) {
    return gradient.dot(direction) < 0;
  }
  
  private List<double[]> minus(List<Delta> a, List<Delta> b) {
    assert(a.size() == b.size());
    for(int i=0;i<a.size();i++) assert a.get(i).layer.equals(b.get(i).layer);
    return ArrayUtil.minus(cvt(a), cvt(b));
  }
  
  private List<double[]> cvt(List<Delta> vector) {
    return vector.stream().map(x -> x.getDelta()).collect(Collectors.toList());
  }
  
  /**
   * Gets min history.
   *
   * @return the min history
   */
  public int getMinHistory() {
    return minHistory;
  }
  
  /**
   * Sets min history.
   *
   * @param minHistory the min history
   * @return the min history
   */
  public LBFGS setMinHistory(int minHistory) {
    this.minHistory = minHistory;
    return this;
  }
  
  /**
   * Gets max history.
   *
   * @return the max history
   */
  public int getMaxHistory() {
    return maxHistory;
  }
  
  /**
   * Sets max history.
   *
   * @param maxHistory the max history
   * @return the max history
   */
  public LBFGS setMaxHistory(int maxHistory) {
    this.maxHistory = maxHistory;
    return this;
  }
}
