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
import com.simiacryptus.mindseye.opt.trainable.Trainable;

import java.util.List;
import java.util.stream.IntStream;

public class SimpleLineSearchCursor implements LineSearchCursor {
  public final Trainable.PointSample origin;
  public final DeltaSet direction;
  public final Trainable subject;
  
  public SimpleLineSearchCursor(Trainable subject, Trainable.PointSample origin, DeltaSet direction) {
    this.origin = origin;
    this.direction = direction;
    this.subject = subject;
  }
  
  protected static double dot(List<DeltaBuffer> a, List<DeltaBuffer> b) {
    assert (a.size() == b.size());
    return IntStream.range(0, a.size()).mapToDouble(i -> a.get(i).dot(b.get(i))).sum();
  }
  
  @Override
  public LineSearchPoint step(double alpha, TrainingMonitor monitor) {
    origin.weights.vector().stream().forEach(d -> d.overwrite());
    direction.vector().stream().forEach(d -> d.write(alpha));
    Trainable.PointSample sample = subject.measure();
    return new LineSearchPoint(sample, dot(direction.vector(), sample.delta.vector()));
  }
  
}
