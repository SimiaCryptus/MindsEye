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

import com.google.gson.JsonObject;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;

/**
 * The basic type of Neural Network LayerBase supporting the backpropigation model of learning. In general, these
 * components define differentiable functions and the accompanying derivatives. The interface is designed to support
 * composability; see DAGNetwork for composition details.
 */
@SuppressWarnings("serial")
public abstract class LayerBase extends RegisteredObjectBase implements Layer {
  
  private final UUID id;
  /**
   * The Frozen.
   */
  protected boolean frozen = false;
  @javax.annotation.Nullable
  private String name;
  
  /**
   * Instantiates a new Nn layer.
   */
  protected LayerBase() {
    id = UUID.randomUUID();
    name = getClass().getSimpleName() + "/" + getId();
  }
  
  /**
   * Instantiates a new Nn layer.
   *
   * @param json the json
   */
  protected LayerBase(@javax.annotation.Nonnull final JsonObject json) {
    if (!getClass().getCanonicalName().equals(json.get("class").getAsString())) {
      throw new IllegalArgumentException(getClass().getCanonicalName() + " != " + json.get("class").getAsString());
    }
    id = UUID.fromString(json.get("id").getAsString());
    if (json.has("isFrozen")) {
      this.frozen = json.get("isFrozen").getAsBoolean();
    }
    if (json.has("name")) {
      setName(json.get("name").getAsString());
    }
  }
  
  /**
   * Instantiates a new Nn layer.
   *
   * @param id   the id
   * @param name the name
   */
  protected LayerBase(final UUID id, final String name) {
    this.id = id;
    this.name = name;
  }
  
  @Override
  public final boolean equals(@Nullable final Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    @Nullable final Layer other = (Layer) obj;
    if (getId() == null) {
      if (other.getId() != null) {
        return false;
      }
    }
    else if (!getId().equals(other.getId())) {
      return false;
    }
    return true;
  }
  
  /**
   * The Id.
   *
   * @return the children
   */
  public List<Layer> getChildren() {
    return Arrays.asList(this);
  }
  
  /**
   * Gets id.
   *
   * @return the id
   */
  @javax.annotation.Nullable
  public Object getId() {
    return id;
  }
  
  /**
   * Gets name.
   *
   * @return the name
   */
  @javax.annotation.Nullable
  public String getName() {
    return name;
  }
  
  /**
   * Sets name.
   *
   * @param name the name
   * @return the name
   */
  @Nonnull
  public Layer setName(final String name) {
    this.name = name;
    return this;
  }
  
  @Override
  public final int hashCode() {
    return getId().hashCode();
  }
  
  /**
   * Is frozen boolean.
   *
   * @return the boolean
   */
  public boolean isFrozen() {
    return frozen;
  }
  
  /**
   * Sets frozen.
   *
   * @param frozen the frozen
   * @return the frozen
   */
  @Nonnull
  public Layer setFrozen(final boolean frozen) {
    this.frozen = frozen;
    return self();
  }
  
  /**
   * Self nn layer.
   *
   * @return the nn layer
   */
  @javax.annotation.Nonnull
  protected final Layer self() {
    return this;
  }
  
  @javax.annotation.Nullable
  @Override
  public final String toString() {
    return getName();
  }
  
  @Override
  protected void _free() {
  
  }
  
}
