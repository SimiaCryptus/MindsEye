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

import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.stream.JsonWriter;
import org.apache.commons.io.IOUtils;

import java.io.*;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/**
 * The basic type of Neural Network Layer supporting the backpropigation model of learning. In general, these components
 * define differentiable functions and the accompanying derivatives. The interface is designed to support composability;
 * see DAGNetwork for composition details.
 */
@SuppressWarnings("serial")
public abstract class NNLayer implements Serializable {
  
  private final UUID id;
  private boolean frozen = false;
  private String name;
  
  /**
   * Instantiates a new Nn layer.
   */
  protected NNLayer() {
    id = UUID.randomUUID();
    name = getClass().getSimpleName() + "/" + getId();
  }
  
  /**
   * Instantiates a new Nn layer.
   *
   * @param json the json
   */
  protected NNLayer(final JsonObject json) {
    if (!getClass().getCanonicalName().equals(json.get("class").getAsString())) {
      throw new IllegalArgumentException(getClass().getCanonicalName() + " != " + json.get("class").getAsString());
    }
    id = UUID.fromString(json.get("id").getAsString());
    if (json.has("isFrozen")) {
      setFrozen(json.get("isFrozen").getAsBoolean());
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
  protected NNLayer(final UUID id, final String name) {
    this.id = id;
    this.name = name;
  }
  
  /**
   * From json nn layer.
   *
   * @param json the json
   * @return the nn layer
   */
  public static NNLayer fromJson(final JsonObject json) { return fromJson(json, null);}
  
  /**
   * From zip nn layer.
   *
   * @param zipfile the zipfile
   * @return the nn layer
   */
  public static NNLayer fromZip(final ZipFile zipfile) {
    Enumeration<? extends ZipEntry> entries = zipfile.entries();
    JsonObject json = null;
    HashMap<String, byte[]> resources = new HashMap<>();
    while (entries.hasMoreElements()) {
      ZipEntry zipEntry = entries.nextElement();
      String name = zipEntry.getName();
      try {
        InputStream inputStream = zipfile.getInputStream(zipEntry);
        if (name.equals("model.json")) {
          json = new GsonBuilder().create().fromJson(new InputStreamReader(inputStream), JsonObject.class);
        }
        else {
          resources.put(name, IOUtils.readFully(inputStream, (int) zipEntry.getSize()));
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
    return fromJson(json, resources);
  }
  
  
  /**
   * From json nn layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the nn layer
   */
  public static NNLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    JsonElement classElement = json.get("class");
    assert null != classElement : json.toString();
    final String className = classElement.getAsString();
    try {
      final Class<?> clazz = Class.forName(className);
      if (null == clazz) throw new ClassNotFoundException(className);
      final Method method = clazz.getMethod("fromJson", JsonObject.class, Map.class);
      if (method.getDeclaringClass() == NNLayer.class) {
        throw new IllegalArgumentException("Cannot find deserialization method for " + className);
      }
      return (NNLayer) method.invoke(null, json, rs);
    } catch (IllegalAccessException | InvocationTargetException | NoSuchMethodException | ClassNotFoundException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * As t.
   *
   * @param <T>         the type parameter
   * @param targetClass the target class
   * @return the t
   */
  @SuppressWarnings("unchecked")
  public <T extends NNLayer> T as(final Class<T> targetClass) {
    HashMap<String, byte[]> resources = new HashMap<>();
    final JsonObject json = getJson(resources, SerialPrecision.Double);
    json.remove("class");
    json.addProperty("class", targetClass.getCanonicalName());
    return (T) NNLayer.fromJson(json, resources);
  }
  
  /**
   * Copy nn layer.
   *
   * @return the nn layer
   */
  public NNLayer copy() {return copy(SerialPrecision.Double);}
  
  /**
   * Copy nn layer.
   *
   * @param precision the precision
   * @return the nn layer
   */
  public NNLayer copy(SerialPrecision precision) {
    HashMap<String, byte[]> resources = new HashMap<>();
    final JsonObject json = getJson(resources, precision);
    return NNLayer.fromJson(json, resources);
  }
  
  @Override
  public final boolean equals(final Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    final NNLayer other = (NNLayer) obj;
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
   * Eval nn result.
   *
   * @param nncontext the nncontext
   * @param array     the array
   * @return the nn result
   */
  public abstract NNResult eval(NNExecutionContext nncontext, NNResult... array);
  
  /**
   * Eval nn result.
   *
   * @param nncontext the nncontext
   * @param array     the array
   * @return the nn result
   */
  public final NNResult eval(final NNExecutionContext nncontext, final Tensor... array) {
    return eval(nncontext, NNConstant.singleResultArray(array));
  }
  
  /**
   * Eval nn result.
   *
   * @param nncontext the nncontext
   * @param array     the array
   * @return the nn result
   */
  public final NNResult eval(final NNExecutionContext nncontext, final Tensor[][] array) {
    return eval(nncontext, NNConstant.singleResultArray(array));
  }
  
  /**
   * Freeze nn layer.
   *
   * @return the nn layer
   */
  public final NNLayer freeze() {
    return setFrozen(true);
  }
  
  /**
   * The Id.
   *
   * @return the children
   */
  public List<NNLayer> getChildren() {
    return Arrays.asList(this);
  }
  
  /**
   * Gets id.
   *
   * @return the id
   */
  public Object getId() {
    return id;
  }
  
  /**
   * Gets json.
   *
   * @param resources      the resources
   * @param dataSerializer the data serializer
   * @return the json
   */
  public abstract JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer);
  
  /**
   * Gets json.
   *
   * @return the json
   */
  public final JsonObject getJson() {
    return getJson(null, SerialPrecision.Double);
  }
  
  /**
   * Write zip.
   *
   * @param out the out
   */
  public final void writeZip(File out) {writeZip(out, SerialPrecision.Double);}
  
  /**
   * Write zip.
   *
   * @param out       the out
   * @param precision the precision
   */
  public final void writeZip(File out, SerialPrecision precision) {
    try (ZipOutputStream zipOutputStream = new ZipOutputStream(new FileOutputStream(out))) {
      writeZip(zipOutputStream, precision);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Write zip.
   *
   * @param out the out
   */
  public final void writeZip(ZipOutputStream out) {writeZip(out, SerialPrecision.Double);}
  
  /**
   * Write zip.
   *
   * @param out       the out
   * @param precision the precision
   */
  public final void writeZip(ZipOutputStream out, SerialPrecision precision) {
    try {
      HashMap<String, byte[]> resources = new HashMap<>();
      JsonObject json = getJson(resources, precision);
      out.putNextEntry(new ZipEntry("model.json"));
      JsonWriter writer = new JsonWriter(new OutputStreamWriter(out));
      writer.setIndent("  ");
      writer.setHtmlSafe(true);
      writer.setSerializeNulls(false);
      new GsonBuilder().setPrettyPrinting().create().toJson(json, writer);
      writer.flush();
      out.closeEntry();
      resources.forEach((name, data) -> {
        try {
          out.putNextEntry(new ZipEntry(name));
          IOUtils.write(data, out);
          out.flush();
          out.closeEntry();
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Gets json string.
   *
   * @return the json string
   */
  public String getJsonString() {
    return new GsonBuilder().setPrettyPrinting().create().toJson(getJson());
  }
  
  /**
   * Gets json stub.
   *
   * @return the json stub
   */
  public JsonObject getJsonStub() {
    final JsonObject json = new JsonObject();
    json.addProperty("class", getClass().getCanonicalName());
    json.addProperty("id", getId().toString());
    json.addProperty("isFrozen", isFrozen());
    json.addProperty("name", getName());
    return json;
  }
  
  /**
   * Gets name.
   *
   * @return the name
   */
  public String getName() {
    return name;
  }
  
  /**
   * Sets name.
   *
   * @param name the name
   * @return the name
   */
  public NNLayer setName(final String name) {
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
  public NNLayer setFrozen(final boolean frozen) {
    this.frozen = frozen;
    return self();
  }
  
  /**
   * Self nn layer.
   *
   * @return the nn layer
   */
  protected final NNLayer self() {
    return this;
  }
  
  /**
   * State list.
   *
   * @return the list
   */
  public abstract List<double[]> state();
  
  @Override
  public final String toString() {
    return getName();
  }
}
