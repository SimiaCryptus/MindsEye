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

package com.simiacryptus.mindseye.labs.matrix;

import com.simiacryptus.mindseye.eval.SampledTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.util.io.NotebookOutput;

/**
 * The interface Optimization strategy.
 */
public interface OptimizationStrategy {
  /**
   * Train validating trainer.
   *
   * @param log               the log
   * @param trainingSubject   the training subject
   * @param validationSubject the validation subject
   * @param monitor           the monitor
   * @return the validating trainer
   */
  ValidatingTrainer train(NotebookOutput log, SampledTrainable trainingSubject, Trainable validationSubject, TrainingMonitor monitor);
}
