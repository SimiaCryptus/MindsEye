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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.BasicTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * An implementation of the Limited-Memory Broyden–Fletcher–Goldfarb–Shanno algorithm
 * https://en.m.wikipedia.org/wiki/Limited-memory_BFGS
 */
public class RLBFGS implements OrientationStrategy<SimpleLineSearchCursor> {
  @Override
  public SimpleLineSearchCursor orient(Trainable subject, PointSample measurement, TrainingMonitor monitor) {
    final DeltaSet<NNLayer> direction = measurement.delta.scale(-1);
    final double magnitude = direction.getMagnitude();
    if (Math.abs(magnitude) < 1e-10) {
      monitor.log(String.format("Zero gradient: %s", magnitude));
    }
    else if (Math.abs(magnitude) < 1e-5) {
      monitor.log(String.format("Low gradient: %s", magnitude));
    }
    StateSet<NNLayer> origin = direction.asState().map(x -> x.backup());
    List<NNLayer> deltaLayers = direction.getMap().entrySet().stream().map(x -> x.getKey()).collect(Collectors.toList());
    double[] weights = new double[direction.getMap().size()];
    NNLayer macroLayer = new NNLayer() {
      NNLayer self = this;
    
      @Override
      public NNResult eval(NNExecutionContext nncontext, NNResult... array) {
        monitor.log(String.format("Recursive Layer Weighting: %s", Arrays.toString(weights)));
        origin.stream().forEach(x -> x.restore());
        IntStream.range(0, deltaLayers.size()).forEach(i -> {
          direction.getMap().get(deltaLayers.get(i)).accumulate(weights[i]);
        });
        PointSample measure = subject.measure(monitor);
        return new NNResult(new Tensor(measure.getMean())) {
          @Override
          public void accumulate(DeltaSet<NNLayer> buffer, TensorList data) {
            buffer.get(self, weights).addInPlace(IntStream.range(0, deltaLayers.size()).mapToDouble(i -> {
              NNLayer layer = deltaLayers.get(i);
              Delta<NNLayer> a = direction.getMap().get(layer);
              Delta<NNLayer> b = measure.delta.getMap().get(layer);
              return b.dot(a);
            }).toArray());
          }
        
          @Override
          public boolean isAlive() {
            return true;
          }
        };
      }
    
      @Override
      public JsonObject getJson() {
        throw new IllegalStateException();
      }
    
      @Override
      public List<double[]> state() {
        return null;
      }
    };
    ArrayTrainable trainable = new ArrayTrainable(new BasicTrainable(macroLayer), new Tensor[][]{{new Tensor()}});
    new IterativeTrainer(trainable)
      .setOrientation(new LBFGS())
      .setLineSearchFactory(n -> new ArmijoWolfeSearch())
      .setMaxIterations(5).run();
    DeltaSet<NNLayer> delta = origin.map(x -> x.copy().backup()).subtract(origin);
    origin.stream().forEach(s -> s.restore());
    return new SimpleLineSearchCursor(subject, measurement, delta).setDirectionType("RLBFGS");
  }
  
  @Override
  public void reset() {
  
  }
}
