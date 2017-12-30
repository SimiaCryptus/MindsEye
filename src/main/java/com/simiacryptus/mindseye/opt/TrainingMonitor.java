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

/**
 * The base class for defining callbacks to monitor training tasks.
 */
public class TrainingMonitor {
  /**
   * Clear.
   */
  public void clear() {
  }
  
  /**
   * This callback intercepts log messages describing the ongoing training process.
   *
   * @param msg the msg
   */
  public void log(final String msg) {
  }
  
  /**
   * This callback is executed periodically, between each line-search process. While processing, the training process is
   * blocked.
   *
   * @param currentPoint the current point
   */
  public void onStepComplete(final Step currentPoint) {
  }
}
