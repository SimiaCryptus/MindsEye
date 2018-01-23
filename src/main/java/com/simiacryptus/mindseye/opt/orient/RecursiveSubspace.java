/*
 * Copyright (c) 2018 by Andrew Charneski.
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
import com.simiacryptus.mindseye.layers.java.PlaceholderLayer;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * An recursive optimization strategy which projects the current space into a reduced-dimensional subspace for a
 * sub-optimization batch apply.
 */
public class RecursiveSubspace implements OrientationStrategy<SimpleLineSearchCursor> {
  
  /**
   * The constant CURSOR_LABEL.
   */
  public static final String CURSOR_LABEL = "RecursiveSubspace";
  private int iterations = 4;
  private double[] weights = null;
  
  @Override
  public SimpleLineSearchCursor orient(Trainable subject, PointSample measurement, TrainingMonitor monitor) {
    PointSample origin = measurement.copyFull().backup();
    NNLayer macroLayer = buildSubspace(subject, measurement, monitor);
    train(monitor, macroLayer);
    macroLayer.eval((NNResult) null);
    DeltaSet<NNLayer> delta = origin.weights.backupCopy().subtract(origin.weights);
    origin.restore();
    return new SimpleLineSearchCursor(subject, origin, delta).setDirectionType(CURSOR_LABEL);
  }
  
  /**
   * Build subspace nn layer.
   *
   * @param subject     the subject
   * @param measurement the measurement
   * @param monitor     the monitor
   * @return the nn layer
   */
  public NNLayer buildSubspace(Trainable subject, PointSample measurement, TrainingMonitor monitor) {
    PointSample origin = measurement.copyFull().backup();
    final DeltaSet<NNLayer> direction = measurement.delta.scale(-1);
    final double magnitude = direction.getMagnitude();
    if (Math.abs(magnitude) < 1e-10) {
      monitor.log(String.format("Zero gradient: %s", magnitude));
    }
    else if (Math.abs(magnitude) < 1e-5) {
      monitor.log(String.format("Low gradient: %s", magnitude));
    }
    boolean hasPlaceholders = direction.getMap().entrySet().stream().filter(x -> x.getKey() instanceof PlaceholderLayer).findAny().isPresent();

    List<NNLayer> deltaLayers = direction.getMap().entrySet().stream().map(x -> x.getKey())
                                         .filter(x -> !(x instanceof PlaceholderLayer))
                                         .collect(Collectors.toList());
    int size = deltaLayers.size() + (hasPlaceholders ? 1 : 0);
    if (null == weights || weights.length != size) weights = new double[size];
    return new NNLayer() {
      NNLayer self = this;
      
      @Override
      public NNResult eval(NNResult... array) {
        origin.restore();
        IntStream.range(0, deltaLayers.size()).forEach(i -> {
          direction.getMap().get(deltaLayers.get(i)).accumulate(weights[hasPlaceholders ? (i + 1) : i]);
        });
        if (hasPlaceholders) {
          direction.getMap().entrySet().stream()
                   .filter(x -> x.getKey() instanceof PlaceholderLayer).distinct()
                   .forEach(entry -> entry.getValue().accumulate(weights[0]));
        }
        PointSample measure = subject.measure(monitor);
        double mean = measure.getMean();
        monitor.log(String.format("RecursiveSubspace: %s <- %s", mean, Arrays.toString(weights)));
        return new NNResult((DeltaSet<NNLayer> buffer, TensorList data) -> {
          DoubleStream deltaStream = deltaLayers.stream().mapToDouble(layer -> {
            Delta<NNLayer> a = direction.getMap().get(layer);
            Delta<NNLayer> b = measure.delta.getMap().get(layer);
            return b.dot(a) / Math.max(Math.sqrt(a.dot(a)), 1e-8);
          });
          if (hasPlaceholders) {
            deltaStream = DoubleStream.concat(DoubleStream.of(
              direction.getMap().keySet().stream().filter(x -> x instanceof PlaceholderLayer).distinct().mapToDouble(layer -> {
                Delta<NNLayer> a = direction.getMap().get(layer);
                Delta<NNLayer> b = measure.delta.getMap().get(layer);
                return b.dot(a) / Math.max(Math.sqrt(a.dot(a)), 1e-8);
              }).sum()), deltaStream);
          }
          buffer.get(self, weights).addInPlace(deltaStream.toArray());
        }, new Tensor(mean)) {
          
          @Override
          public boolean isAlive() {
            return true;
          }
        };
      }
      
      @Override
      public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
        throw new IllegalStateException();
      }
      
      @Override
      public List<double[]> state() {
        return null;
      }
    };
  }
  
  /**
   * Train.
   *
   * @param monitor    the monitor
   * @param macroLayer the macro layer
   */
  public void train(TrainingMonitor monitor, NNLayer macroLayer) {
    ArrayTrainable trainable = new ArrayTrainable(new BasicTrainable(macroLayer), new Tensor[][]{{new Tensor()}});
    new IterativeTrainer(trainable)
      .setOrientation(new LBFGS())
      .setLineSearchFactory(n -> new ArmijoWolfeSearch())
      .setMonitor(new TrainingMonitor() {
        @Override
        public void log(String msg) {
          monitor.log("\t" + msg);
        }
      })
      .setMaxIterations(getIterations()).setIterationsPerSample(getIterations()).run();
  }
  
  @Override
  public void reset() {
    weights = null;
  }
  
  /**
   * Gets iterations.
   *
   * @return the iterations
   */
  public int getIterations() {
    return iterations;
  }
  
  /**
   * Sets iterations.
   *
   * @param iterations the iterations
   * @return the iterations
   */
  public RecursiveSubspace setIterations(int iterations) {
    this.iterations = iterations;
    return this;
  }
}
