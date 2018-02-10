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

import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;

import javax.annotation.Nonnull;

/**
 * The most basic type of orientation, which uses the raw function gradient.
 */
public class GradientDescent extends OrientationStrategyBase<SimpleLineSearchCursor> {
  
  @javax.annotation.Nonnull
  @Override
  public SimpleLineSearchCursor orient(final Trainable subject, @javax.annotation.Nonnull final PointSample measurement, @javax.annotation.Nonnull final TrainingMonitor monitor) {
    @Nonnull final DeltaSet<NNLayer> direction = measurement.delta.scale(-1);
    final double magnitude = direction.getMagnitude();
    if (Math.abs(magnitude) < 1e-10) {
      monitor.log(String.format("Zero gradient: %s", magnitude));
    }
    else if (Math.abs(magnitude) < 1e-5) {
      monitor.log(String.format("Low gradient: %s", magnitude));
    }
    @javax.annotation.Nonnull SimpleLineSearchCursor gd = new SimpleLineSearchCursor(subject, measurement, direction).setDirectionType("GD");
    direction.freeRef();
    return gd;
  }
  
  @Override
  public void reset() {
  
  }
  
  @Override
  protected void _free() {
  
  }
}
