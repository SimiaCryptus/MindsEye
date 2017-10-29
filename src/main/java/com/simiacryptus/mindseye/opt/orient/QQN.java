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
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchPoint;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;

/**
 * Quadratic Quasi-Newton strategy
 */
public class QQN extends LBFGS {
  
  @Override
  public LineSearchCursor orient(Trainable subject, PointSample origin, TrainingMonitor monitor) {
    addToHistory(origin, monitor);
    SimpleLineSearchCursor lbfgsCursor = _orient(subject, origin, monitor);
    final DeltaSet lbfgs = lbfgsCursor.direction;
    DeltaSet gd = origin.delta.scale(-1.0 / origin.count);
    double lbfgsMag = lbfgs.getMagnitude();
    double gdMag = gd.getMagnitude();
    if ((Math.abs(lbfgsMag - gdMag) / (lbfgsMag + gdMag)) > 1e-2) {
      DeltaSet scaledGradient = gd.scale(lbfgsMag / gdMag);
      monitor.log(String.format("Returning Quadratic Cursor"));
      return new LineSearchCursor() {

        @Override
        public String getDirectionType() {
          return "QQN";
        }

        @Override
        public LineSearchPoint step(double t, TrainingMonitor monitor) {
          if (!Double.isFinite(t)) throw new IllegalArgumentException();
          reset();
          position(t).accumulate();
          PointSample sample = subject.measure(true).setRate(t);
          //monitor.log(String.format("delta buffers %d %d %d %d %d", sample.delta.map.size(), origin.delta.map.size(), lbfgs.map.size(), gd.map.size(), scaledGradient.map.size()));
          addToHistory(sample, monitor);
          DeltaSet tangent = scaledGradient.scale(1 - 2 * t).add(lbfgs.scale(2 * t));
          return new LineSearchPoint(sample, SimpleLineSearchCursor.dot(tangent.vector(), sample.delta.vector()));
        }

        @Override
        public DeltaSet position(double t) {
          if (!Double.isFinite(t)) throw new IllegalArgumentException();
          return scaledGradient.scale(t - t * t).add(lbfgs.scale(t * t));
        }
        
        @Override
        public void reset() {
          lbfgsCursor.reset();
        }
      };
    }
    else {
      return lbfgsCursor;
    }
  }
  
  
}
