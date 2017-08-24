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

import com.simiacryptus.mindseye.layers.Delta;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchPoint;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.mindseye.opt.trainable.Trainable;
import com.simiacryptus.mindseye.opt.trainable.Trainable.PointSample;

import java.util.List;

/**
 * Quadratic Quasi-Newton strategy
 */
public class QQN extends LBFGS {
  
  @Override
  public LineSearchCursor orient(Trainable subject, PointSample origin, TrainingMonitor monitor) {
    addToHistory(origin, monitor);
    SimpleLineSearchCursor lbfgs = _orient(subject, origin, monitor);
    double lbfgsMag = lbfgs.direction.getMagnitude();
    DeltaSet gd = origin.delta.scale(-1);
    double gdMag = gd.getMagnitude();
    if((Math.abs(lbfgsMag - gdMag) / (lbfgsMag + gdMag)) > 1e-2) {
      return new LineSearchCursor(){
  
        @Override
        public String getDirectionType() {
          return "QQN";
        }
  
        @Override
        public LineSearchPoint step(double t, TrainingMonitor monitor) {
          if(!Double.isFinite(t)) throw new IllegalArgumentException();
          position(t).write();
          PointSample sample = measure(t, monitor);
          List<Delta> tangent = gd.scale(1-2*t).add(lbfgs.direction.scale(2*t)).vector();
          return new LineSearchPoint(sample, SimpleLineSearchCursor.dot(tangent, sample.delta.vector()));
        }
  
        @Override
        public DeltaSet position(double t) {
          if(!Double.isFinite(t)) throw new IllegalArgumentException();
          return gd.scale(t-t*t).add(lbfgs.direction.scale(t*t));
        }
  
        @Override
        public PointSample measure(double t, TrainingMonitor monitor) {
          PointSample sample = subject.measure().setRate(t);
          addToHistory(sample, monitor);
          return sample;
        }
        
        @Override
        public void reset() {
          lbfgs.reset();
        }
      };
    } else {
      return lbfgs;
    }
  }
  
  
}
