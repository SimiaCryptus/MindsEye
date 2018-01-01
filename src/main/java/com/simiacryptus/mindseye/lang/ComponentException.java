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
 * An exception occured within a NNLayer component. This is most often due to invalid input or configuration.
 */
@SuppressWarnings("serial")
public class ComponentException extends RuntimeException {
  
  /**
   * Instantiates a new Component exception.
   */
  public ComponentException() {
  }
  
  /**
   * Instantiates a new Component exception.
   *
   * @param message the message
   */
  public ComponentException(final String message) {
    super(message);
  }
  
  /**
   * Instantiates a new Component exception.
   *
   * @param message the message
   * @param cause   the cause
   */
  public ComponentException(final String message, final Throwable cause) {
    super(message, cause);
  }
  
  /**
   * Instantiates a new Component exception.
   *
   * @param message            the message
   * @param cause              the cause
   * @param enableSuppression  the enable suppression
   * @param writableStackTrace the writable stack trace
   */
  public ComponentException(final String message, final Throwable cause, final boolean enableSuppression, final boolean writableStackTrace) {
    super(message, cause, enableSuppression, writableStackTrace);
  }
  
  /**
   * Instantiates a new Component exception.
   *
   * @param cause the cause
   */
  public ComponentException(final Throwable cause) {
    super(cause);
  }
}
