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
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.java.PlaceholderLayer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchPoint;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.util.ArrayUtil;

import java.util.*;
import java.util.stream.Collectors;

/**
 * An implementation of the Limited-Memory Broyden–Fletcher–Goldfarb–Shanno algorithm
 * https://en.m.wikipedia.org/wiki/Limited-memory_BFGS
 */
public class LBFGS implements OrientationStrategy<SimpleLineSearchCursor> {
  
  /**
   * The History.
   */
  public final TreeSet<PointSample> history = new TreeSet<>(Comparator.comparing(x -> -x.getMean()));
  /**
   * The Verbose.
   */
  protected boolean verbose = false;
  private int maxHistory = 30;
  private int minHistory = 3;
  
  private static boolean isFinite(final DoubleBufferSet<?, ?> delta) {
    return delta.stream().parallel().flatMapToDouble(y -> Arrays.stream(y.getDelta())).allMatch(d -> Double.isFinite(d));
  }
  
  /**
   * Accept boolean.
   *
   * @param gradient  the gradient
   * @param direction the direction
   * @return the boolean
   */
  protected boolean accept(final DeltaSet<NNLayer> gradient, final DeltaSet<NNLayer> direction) {
    return gradient.dot(direction) < 0;
  }
  
  /**
   * Add to history.
   *
   * @param measurement the measurement
   * @param monitor     the monitor
   */
  public void addToHistory(final PointSample measurement, final TrainingMonitor monitor) {
    final PointSample copyFull = measurement.copyFull();
    if (!LBFGS.isFinite(copyFull.delta)) {
      if (verbose) {
        monitor.log("Corrupt measurement");
      }
    }
    else if (!LBFGS.isFinite(copyFull.weights)) {
      if (verbose) {
        monitor.log("Corrupt measurement");
      }
    }
    else if (history.isEmpty() || !history.stream().filter(x -> x.sum <= copyFull.sum).findAny().isPresent()) {
      if (verbose) {
        monitor.log(String.format("Adding measurement %s to history. Total: %s", Long.toHexString(System.identityHashCode(copyFull)), history.size()));
      }
      history.add(copyFull);
    }
  }
  
  private SimpleLineSearchCursor cursor(final Trainable subject, final PointSample measurement, final String type, final DeltaSet<NNLayer> result) {
    return new SimpleLineSearchCursor(subject, measurement, result) {
      @Override
      public LineSearchPoint step(final double t, final TrainingMonitor monitor) {
        final LineSearchPoint measure = super.step(t, monitor);
        addToHistory(measure.point, monitor);
        return measure;
      }
    }
      .setDirectionType(type);
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
  public LBFGS setMaxHistory(final int maxHistory) {
    this.maxHistory = maxHistory;
    return this;
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
  public LBFGS setMinHistory(final int minHistory) {
    this.minHistory = minHistory;
    return this;
  }
  
  /**
   * Lbfgs delta set.
   *
   * @param measurement the measurement
   * @param monitor     the monitor
   * @param history     the history
   * @return the delta set
   */
  protected DeltaSet<NNLayer> lbfgs(final PointSample measurement, final TrainingMonitor monitor, final List<PointSample> history) {
    final DeltaSet<NNLayer> result = measurement.delta.scale(-1);
    if (history.size() > minHistory) {
      if (!step(measurement, monitor, history, result.copy())) {
        monitor.log("Orientation rejected. Popping history element from " + history.stream().map(x -> String.format("%s", x.getMean())).reduce((a, b) -> a + ", " + b).get());
        return lbfgs(measurement, monitor, history.subList(0, history.size() - 1));
      }
      else {
        this.history.clear();
        this.history.addAll(history);
        return result;
      }
    }
    else {
      monitor.log(String.format("LBFGS Accumulation History: %s points", history.size()));
      return null;
    }
  }
  
  private boolean step(PointSample measurement, TrainingMonitor monitor, List<PointSample> history, DeltaSet<NNLayer> original) {
    try {
      //final DeltaSet<NNLayer> original = result.copy();
      DeltaSet<NNLayer> p = measurement.delta.copy();
      if ((!p.stream().parallel().allMatch(y -> Arrays.stream(y.getDelta()).allMatch(d -> Double.isFinite(d))))) {
        throw new IllegalStateException();
      }
      final double[] alphas = new double[history.size()];
      for (int i = history.size() - 2; i >= 0; i--) {
        final DeltaSet<NNLayer> sd = history.get(i + 1).weights.subtract(history.get(i).weights);
        final DeltaSet<NNLayer> yd = history.get(i + 1).delta.subtract(history.get(i).delta);
        final double denominator = sd.dot(yd);
        if (0 == denominator) {
          monitor.log("Orientation vanished.");
          return false;
        }
        alphas[i] = p.dot(sd) / denominator;
        p = p.subtract(yd.scale(alphas[i]));
        if ((!p.stream().parallel().allMatch(y -> Arrays.stream(y.getDelta()).allMatch(d -> Double.isFinite(d))))) {
          throw new IllegalStateException();
        }
      }
      final DeltaSet<NNLayer> sk = history.get(history.size() - 1).weights.subtract(history.get(history.size() - 2).weights);
      final DeltaSet<NNLayer> yk = history.get(history.size() - 1).delta.subtract(history.get(history.size() - 2).delta);
      p = p.scale(sk.dot(yk) / yk.dot(yk));
      if ((!p.stream().parallel().allMatch(y -> Arrays.stream(y.getDelta()).allMatch(d -> Double.isFinite(d))))) {
        throw new IllegalStateException();
      }
      for (int i = 0; i < history.size() - 1; i++) {
        final DeltaSet<NNLayer> sd = history.get(i + 1).weights.subtract(history.get(i).weights);
        final DeltaSet<NNLayer> yd = history.get(i + 1).delta.subtract(history.get(i).delta);
        final double beta = p.dot(yd) / sd.dot(yd);
        p = p.add(sd.scale(alphas[i] - beta));
        if ((!p.stream().parallel().allMatch(y -> Arrays.stream(y.getDelta()).allMatch(d -> Double.isFinite(d))))) {
          throw new IllegalStateException();
        }
      }
      for (final Map.Entry<NNLayer, Delta<NNLayer>> e : original.getMap().entrySet()) {
        final double[] delta = p.getMap().get(e.getKey()).getDelta();
        Arrays.setAll(e.getValue().getDelta(), j -> delta[j]);
      }
      final double mag = Math.sqrt(original.dot(original));
      final double magGrad = Math.sqrt(original.dot(original));
      final double dot = original.dot(original) / (mag * magGrad);
      final List<String> anglesPerLayer = measurement.delta.getMap().entrySet().stream()
        .filter(e -> !(e.getKey() instanceof PlaceholderLayer)) // This would be too verbose
        .map((final Map.Entry<NNLayer, Delta<NNLayer>> e) -> {
          final double[] lbfgsVector = original.getMap().get(e.getKey()).getDelta();
          for (int index = 0; index < lbfgsVector.length; index++) {
            lbfgsVector[index] = Double.isFinite(lbfgsVector[index]) ? lbfgsVector[index] : 0;
          }
          final double[] gradientVector = original.getMap().get(e.getKey()).getDelta();
          for (int index = 0; index < gradientVector.length; index++) {
            gradientVector[index] = Double.isFinite(gradientVector[index]) ? gradientVector[index] : 0;
          }
          final double lbfgsMagnitude = ArrayUtil.magnitude(lbfgsVector);
          final double gradientMagnitude = ArrayUtil.magnitude(gradientVector);
          if (!Double.isFinite(gradientMagnitude)) throw new IllegalStateException();
          if (!Double.isFinite(lbfgsMagnitude)) throw new IllegalStateException();
          final String layerName = measurement.delta.getMap().get(e.getKey()).layer.getName();
          if (gradientMagnitude == 0.0) {
            return String.format("%s = %.3e", layerName, lbfgsMagnitude);
          }
          else {
            final double dotP = ArrayUtil.dot(lbfgsVector, gradientVector) / (lbfgsMagnitude * gradientMagnitude);
            return String.format("%s = %.3f/%.3e", layerName, dotP, lbfgsMagnitude / gradientMagnitude);
          }
        }).collect(Collectors.toList());
      monitor.log(String.format("LBFGS Orientation magnitude: %.3e, gradient %.3e, dot %.3f; %s", mag, magGrad, dot, anglesPerLayer));
      return accept(measurement.delta, original);
    } catch (Throwable e) {
      monitor.log(String.format("LBFGS Orientation Error: %s", e.getMessage()));
      return false;
    }
  }
  
  @Override
  public SimpleLineSearchCursor orient(final Trainable subject, final PointSample measurement, final TrainingMonitor monitor) {
    addToHistory(measurement, monitor);
    final List<PointSample> history = Arrays.asList(this.history.toArray(new PointSample[]{}));
    final DeltaSet<NNLayer> result = lbfgs(measurement, monitor, history);
    SimpleLineSearchCursor returnValue;
    if (null == result) {
      returnValue = cursor(subject, measurement, "GD", measurement.delta.scale(-1));
    }
    else {
      returnValue = cursor(subject, measurement, "LBFGS", result);
    }
    while (this.history.size() > (null == result ? minHistory : maxHistory)) {
      final PointSample remove = this.history.pollFirst();
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
}
