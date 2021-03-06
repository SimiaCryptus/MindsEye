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
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.apache.commons.io.IOUtils;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.*;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/**
 * The interface Layer.
 */
public interface Layer extends ReferenceCounting, Serializable {

  /**
   * From json nn key.
   *
   * @param json the json
   * @return the nn key
   */
  @Nonnull
  static Layer fromJson(@Nonnull final JsonObject json) {
    return fromJson(json, null);
  }

  /**
   * From zip nn key.
   *
   * @param zipfile the zipfile
   * @return the nn key
   */
  @Nonnull
  static Layer fromZip(@Nonnull final ZipFile zipfile) {
    Enumeration<? extends ZipEntry> entries = zipfile.entries();
    @Nullable JsonObject json = null;
    @Nonnull HashMap<CharSequence, byte[]> resources = new HashMap<>();
    while (entries.hasMoreElements()) {
      ZipEntry zipEntry = entries.nextElement();
      CharSequence name = zipEntry.getName();
      try {
        InputStream inputStream = zipfile.getInputStream(zipEntry);
        if (name.equals("model.json")) {
          json = new GsonBuilder().create().fromJson(new InputStreamReader(inputStream), JsonObject.class);
        } else {
          resources.put(name, IOUtils.readFully(inputStream, (int) zipEntry.getSize()));
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
    return fromJson(json, resources);
  }


  /**
   * From json nn key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the nn key
   */
  @Nonnull
  static Layer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    JsonElement classElement = json.get("class");
    assert null != classElement : json.toString();
    final String className = classElement.getAsString();
    try {
      final Class<?> clazz = Class.forName(className);
      if (null == clazz) throw new ClassNotFoundException(className);
      final Method method = clazz.getMethod("fromJson", JsonObject.class, Map.class);
      if (method.getDeclaringClass() == Layer.class) {
        throw new IllegalArgumentException("Cannot find deserialization method for " + className);
      }
      @Nonnull Layer invoke = (Layer) method.invoke(null, json, rs);
      if (null == invoke) throw new IllegalStateException();
      return invoke;
    } catch (@Nonnull IllegalAccessException | InvocationTargetException | NoSuchMethodException | ClassNotFoundException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * And then key.
   *
   * @param append the append
   * @return the key
   */
  default PipelineNetwork andThen(Layer append) {
    return PipelineNetwork.build(1,
        this,
        append
    );
  }

  /**
   * Free and then pipeline network.
   *
   * @param append the append
   * @return the pipeline network
   */
  default PipelineNetwork freeAndThen(Layer append) {
    PipelineNetwork build = andThen(append);
    this.freeRef();
    return build;
  }

  /**
   * And then wrap pipeline network.
   *
   * @param append the append
   * @return the pipeline network
   */
  default PipelineNetwork andThenWrap(Layer append) {
    assert append.assertAlive();
    assert assertAlive();
    PipelineNetwork wrap = PipelineNetwork.build(1,
        this,
        append
    );
    append.freeRef();
    return wrap;
  }

  /**
   * Free and then wrap pipeline network.
   *
   * @param append the append
   * @return the pipeline network
   */
  default PipelineNetwork freeAndThenWrap(Layer append) {
    assert append.assertAlive();
    PipelineNetwork build = PipelineNetwork.wrap(1,
        this,
        append
    );
    return build;
  }

  /**
   * As t.
   *
   * @param <T>         the type parameter
   * @param targetClass the target class
   * @return the t
   */
  @Nonnull
  @SuppressWarnings("unchecked")
  default <T extends Layer> T as(@Nonnull final Class<T> targetClass) {
    @Nonnull HashMap<CharSequence, byte[]> resources = new HashMap<>();
    final JsonObject json = getJson(resources, SerialPrecision.Double);
    json.remove("class");
    json.addProperty("class", targetClass.getCanonicalName());
    return (T) fromJson(json, resources);
  }

  /**
   * Copy nn key.
   *
   * @return the nn key
   */
  @Nonnull
  default Layer copy() {
    return copy(SerialPrecision.Double);
  }

  /**
   * Copy nn key.
   *
   * @param precision the precision
   * @return the nn key
   */
  @Nonnull
  default Layer copy(SerialPrecision precision) {
    assertAlive();
    @Nonnull HashMap<CharSequence, byte[]> resources = new HashMap<>();
    final JsonObject json = getJson(resources, precision);
    return Layer.fromJson(json, resources);
  }

  /**
   * Eval nn result.
   *
   * @param array the array
   * @return the nn result
   */
  @Nullable
  default Result eval(Result... array) {
    Arrays.stream(array).forEach(ReferenceCounting::addRef);
    Arrays.stream(array).map(Result::getData).forEach(ReferenceCounting::addRef);
    return evalAndFree(array);
  }

  /**
   * Eval and free nn result.
   *
   * @param array the array
   * @return the nn result
   */
  @Nullable
  default Result evalAndFree(Result... array) {
    Result result = eval(array);
    Arrays.stream(array).map(Result::getData).forEach(ReferenceCounting::freeRef);
    Arrays.stream(array).forEach(ReferenceCounting::freeRef);
    return result;
  }

  /**
   * Eval nn result.
   *
   * @param array the array
   * @return the nn result
   */
  @Nullable
  default Result eval(@Nonnull final Tensor... array) {
    Result[] input = ConstantResult.singleResultArray(array);
    Result eval = eval(input);
    Arrays.stream(input).forEach(ReferenceCounting::freeRef);
    Arrays.stream(input).map(Result::getData).forEach(ReferenceCounting::freeRef);
    return eval;
  }

  /**
   * Eval nn result.
   *
   * @param array the array
   * @return the nn result
   */
  @Nullable
  default Result eval(@Nonnull final Tensor[][] array) {
    Result[] input = ConstantResult.singleResultArray(array);
    Result eval = eval(input);
    Arrays.stream(input).forEach(ReferenceCounting::freeRef);
    Arrays.stream(input).map(Result::getData).forEach(ReferenceCounting::freeRef);
    return eval;
  }

  /**
   * Freeze nn key.
   *
   * @return the nn key
   */
  @Nonnull
  default Layer freeze() {
    return setFrozen(true);
  }

  /**
   * The Id.
   *
   * @return the children
   */
  List<Layer> getChildren();

  /**
   * Gets id.
   *
   * @return the id
   */
  @Nullable
  UUID getId();

  /**
   * Gets json.
   *
   * @param resources      the resources
   * @param dataSerializer the data serializer
   * @return the json
   */
  JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer);

  /**
   * Gets json.
   *
   * @return the json
   */
  default JsonObject getJson() {
    return getJson(null, SerialPrecision.Double);
  }

  /**
   * Write zip.
   *
   * @param out the out
   */
  default void writeZip(@Nonnull File out) {
    writeZip(out, SerialPrecision.Double);
  }

  /**
   * Write zip.
   *
   * @param out       the out
   * @param precision the precision
   */
  default void writeZip(@Nonnull File out, SerialPrecision precision) {
    try (@Nonnull ZipOutputStream zipOutputStream = new ZipOutputStream(new FileOutputStream(out))) {
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
  default void writeZip(@Nonnull ZipOutputStream out) {
    writeZip(out, SerialPrecision.Double);
  }

  /**
   * Write zip.
   *
   * @param out       the out
   * @param precision the precision
   */
  default void writeZip(@Nonnull ZipOutputStream out, SerialPrecision precision) {
    try {
      @Nonnull HashMap<CharSequence, byte[]> resources = new HashMap<>();
      JsonObject json = getJson(resources, precision);
      out.putNextEntry(new ZipEntry("model.json"));
      @Nonnull JsonWriter writer = new JsonWriter(new OutputStreamWriter(out));
      writer.setIndent("  ");
      writer.setHtmlSafe(true);
      writer.setSerializeNulls(false);
      new GsonBuilder().setPrettyPrinting().create().toJson(json, writer);
      writer.flush();
      out.closeEntry();
      resources.forEach((name, data) -> {
        try {
          out.putNextEntry(new ZipEntry(String.valueOf(name)));
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
  default CharSequence getJsonString() {
    return new GsonBuilder().setPrettyPrinting().create().toJson(getJson());
  }

  /**
   * Gets json stub.
   *
   * @return the json stub
   */
  @Nonnull
  default JsonObject getJsonStub() {
    assertAlive();
    @Nonnull final JsonObject json = new JsonObject();
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
  @Nullable
  String getName();

  /**
   * Sets name.
   *
   * @param name the name
   * @return the name
   */
  @Nonnull
  Layer setName(final String name);

  /**
   * Is frozen boolean.
   *
   * @return the boolean
   */
  boolean isFrozen();

  /**
   * Sets frozen.
   *
   * @param frozen the frozen
   * @return the frozen
   */
  @Nonnull
  Layer setFrozen(final boolean frozen);

  /**
   * State list.
   *
   * @return the list
   */
  @Nullable
  List<double[]> state();

  /**
   * Copy and free key.
   *
   * @return the key
   */
  default Layer copyAndFree() {
    Layer copy = copy();
    freeRef();
    return copy;
  }
}
