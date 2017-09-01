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

package com.simiacryptus.mindseye.layers.cudnn;

/**
 * The type Device out of memory error.
 */
public final class DeviceOutOfMemoryError extends RuntimeException {
  /**
   * Instantiates a new Device out of memory error.
   *
   * @param msg   the msg
   * @param cause the cause
   */
  public DeviceOutOfMemoryError(String msg, Exception cause) {
    super(msg, cause);
  }
  
  /**
   * Instantiates a new Device out of memory error.
   *
   * @param message the message
   */
  public DeviceOutOfMemoryError(String message) {
    super(message);
  }
}
