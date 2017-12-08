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

package com.simiacryptus.mindseye.test;

import java.awt.*;
import java.util.List;

/**
 * The type Problem run.
 */
public class ProblemRun {
  /**
   * The History.
   */
  public final List<StepRecord> history;
  /**
   * The Name.
   */
  public final String name;
  /**
   * The Color.
   */
  public final Color color;
  
  /**
   * Instantiates a new Problem run.
   *
   * @param name    the name
   * @param color   the color
   * @param history the history
   */
  public ProblemRun(String name, Color color, List<StepRecord> history) {
    this.history = history;
    this.name = name;
    this.color = color;
  }
}
