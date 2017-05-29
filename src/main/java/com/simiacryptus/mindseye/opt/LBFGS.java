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

package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.layers.DeltaBuffer;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.mindseye.opt.trainable.Trainable;
import com.simiacryptus.mindseye.opt.trainable.Trainable.PointSample;
import com.simiacryptus.util.ArrayUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static com.simiacryptus.util.ArrayUtil.add;
import static com.simiacryptus.util.ArrayUtil.dot;

public class LBFGS implements OrientationStrategy {
  
  public final ArrayList<PointSample> history = new ArrayList<>();
  private int minHistory = 3;
  private int maxHistory = 10;
  
  @Override
  public LineSearchCursor orient(Trainable subject, PointSample measurement, TrainingMonitor monitor) {
    if (!measurement.delta.vector().stream().allMatch(y -> Arrays.stream(y.delta).allMatch(d -> Double.isFinite(d)))) {
      monitor.log("Corrupt measurement");
      return new SimpleLineSearchCursor(subject, measurement, DeltaSet.fromList(measurement.delta.vector().stream().map(x -> x.scale(-1)).collect(Collectors.toList()))
      );
    }
    if (!measurement.weights.vector().stream().allMatch(y -> Arrays.stream(y.delta).allMatch(d -> Double.isFinite(d)))) {
      monitor.log("Corrupt measurement");
      return new SimpleLineSearchCursor(subject, measurement, DeltaSet.fromList(measurement.delta.vector().stream().map(x -> x.scale(-1)).collect(Collectors.toList()))
      );
    }
    if(history.isEmpty() || measurement.value != history.get(history.size()-1).value) history.add(measurement);
    while(history.size() > maxHistory) history.remove(0);
    return new SimpleLineSearchCursor(subject, measurement, _orient(measurement, monitor));
  }
  
  public DeltaSet _orient(PointSample measurement, TrainingMonitor monitor) {
    List<DeltaBuffer> defaultValue = measurement.delta.vector().stream().map(x -> x.scale(-1)).collect(Collectors.toList());
    
    // See also https://papers.nips.cc/paper/5333-large-scale-l-bfgs-using-mapreduce (Large-scale L-BFGS using MapReduce)
    List<DeltaBuffer> descent = defaultValue;
    if (history.size() > minHistory) {
      List<double[]> p = descent.stream().map(x -> x.copyDelta()).collect(Collectors.toList());
      assert (p.stream().allMatch(y -> Arrays.stream(y).allMatch(d -> Double.isFinite(d))));
      double[] alphas = new double[history.size()];
      for (int i = history.size() - 2; i >= 0; i--) {
        List<double[]> si = minus((history.get(i + 1).weights.vector()), history.get(i).weights.vector());
        List<double[]> yi = minus(history.get(i + 1).delta.vector(), history.get(i).delta.vector());
        double denominator = dot(si, yi);
        if (0 == denominator) {
          history.remove(0);
          monitor.log("Orientation vanished. Popping history element from " + (history.size() - 1));
          return _orient(measurement, monitor);
        }
        alphas[i] = dot(si, p) / denominator;
        p = ArrayUtil.minus(p, ArrayUtil.multiply(yi, alphas[i]));
        assert (p.stream().allMatch(y -> Arrays.stream(y).allMatch(d -> Double.isFinite(d))));
      }
      List<double[]> sk1 = minus(history.get(history.size() - 1).weights.vector(), history.get(history.size() - 2).weights.vector());
      List<double[]> yk1 = minus(history.get(history.size() - 1).delta.vector(), history.get(history.size() - 2).delta.vector());
      p = ArrayUtil.multiply(p, dot(sk1, yk1) / dot(yk1, yk1));
      assert (p.stream().allMatch(y -> Arrays.stream(y).allMatch(d -> Double.isFinite(d))));
      for (int i = 0; i < history.size() - 1; i++) {
        List<double[]> si = minus(history.get(i + 1).weights.vector(), history.get(i).weights.vector());
        List<double[]> yi = minus(history.get(i + 1).delta.vector(), history.get(i).delta.vector());
        double beta = dot(yi, p) / dot(si, yi);
        p = add(p, ArrayUtil.multiply(si, alphas[i] - beta));
        assert (p.stream().allMatch(y -> Arrays.stream(y).allMatch(d -> Double.isFinite(d))));
      }
      List<double[]> _p = p;
      for (int i = 0; i < descent.size(); i++) {
        int _i = i;
        Arrays.setAll(descent.get(i).delta, j -> _p.get(_i)[j]);
      }
    }
    if (accept(measurement.delta.vector(), descent)) {
      history.remove(0);
      monitor.log("Orientation rejected. Popping history element from " + (history.size() - 1));
      return _orient(measurement, monitor);
    }
    return DeltaSet.fromList(descent);
  }
  
  private boolean accept(List<DeltaBuffer> gradient, List<DeltaBuffer> direction) {
    return dot(cvt(gradient), cvt(direction)) > 0;
  }
  
  private List<double[]> minus(List<DeltaBuffer> a, List<DeltaBuffer> b) {
    return ArrayUtil.minus(cvt(a), cvt(b));
  }
  
  private List<double[]> cvt(List<DeltaBuffer> vector) {
    return vector.stream().map(x -> x.delta).collect(Collectors.toList());
  }
  
  public int getMinHistory() {
    return minHistory;
  }
  
  public LBFGS setMinHistory(int minHistory) {
    this.minHistory = minHistory;
    return this;
  }
  
  public int getMaxHistory() {
    return maxHistory;
  }
  
  public LBFGS setMaxHistory(int maxHistory) {
    this.maxHistory = maxHistory;
    return this;
  }
}
