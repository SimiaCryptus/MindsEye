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

package com.simiacryptus.mindseye.lang;

/**
 * A custom type for OOM errors so we can track an localCopy exception. In the case of a GPU OOM exception, we will
 * likely have an interesting GpuError. For a java OOM, we will wrap it anyway so we have a consolidated exception
 * class.
 */
@SuppressWarnings("serial")
public class OutOfMemoryError extends RuntimeException {
  
  /**
   * Instantiates a new Out of memory error.
   */
  public OutOfMemoryError() {
  }
  
  /**
   * Instantiates a new Out of memory error.
   *
   * @param message the message
   */
  public OutOfMemoryError(final String message) {
    super(message);
  }
  
  /**
   * Instantiates a new Out of memory error.
   *
   * @param message the message
   * @param cause   the cause
   */
  public OutOfMemoryError(final String message, final Throwable cause) {
    super(message, cause);
  }
  
  /**
   * Instantiates a new Out of memory error.
   *
   * @param message            the message
   * @param cause              the cause
   * @param enableSuppression  the enable suppression
   * @param writableStackTrace the writable stack trace
   */
  public OutOfMemoryError(final String message, final Throwable cause, final boolean enableSuppression, final boolean writableStackTrace) {
    super(message, cause, enableSuppression, writableStackTrace);
  }
  
  /**
   * Instantiates a new Out of memory error.
   *
   * @param cause the cause
   */
  public OutOfMemoryError(final Throwable cause) {
    super(cause);
  }
}
