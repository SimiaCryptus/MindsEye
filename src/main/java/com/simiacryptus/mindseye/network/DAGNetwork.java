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

package com.simiacryptus.mindseye.network;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.java.WrapperLayer;
import com.simiacryptus.util.MonitoredItem;
import com.simiacryptus.util.MonitoredObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Directed Acyclical Graph Network The base class for all conventional network wiring.
 */
@SuppressWarnings("serial")
public abstract class DAGNetwork extends LayerBase {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DAGNetwork.class);
  /**
   * The Input handles.
   */
  public final List<UUID> inputHandles = new ArrayList<>();
  /**
   * The Input nodes.
   */
  public final LinkedHashMap<UUID, InputNode> inputNodes = new LinkedHashMap<>();
  /**
   * The Labels.
   */
  protected final LinkedHashMap<String, UUID> labels = new LinkedHashMap<>();
  /**
   * The Layers by id.
   */
  protected final LinkedHashMap<Object, Layer> layersById = new LinkedHashMap<>();
  /**
   * The Nodes by id.
   */
  protected final LinkedHashMap<UUID, DAGNode> nodesById = new LinkedHashMap<>();
  
  /**
   * Instantiates a new Dag network.
   *
   * @param inputs the inputs
   */
  public DAGNetwork(final int inputs) {
    for (int i = 0; i < inputs; i++) {
      addInput();
    }
  }
  
  /**
   * Instantiates a new Dag network.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected DAGNetwork(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    super(json);
    for (@javax.annotation.Nonnull final JsonElement item : json.getAsJsonArray("inputs")) {
      @javax.annotation.Nonnull final UUID key = UUID.fromString(item.getAsString());
      inputHandles.add(key);
      InputNode replaced = inputNodes.put(key, new InputNode(this, key));
      if (null != replaced) replaced.freeRef();
    }
    final JsonObject jsonNodes = json.getAsJsonObject("nodes");
    final JsonObject jsonLayers = json.getAsJsonObject("layers");
    final JsonObject jsonLinks = json.getAsJsonObject("links");
    final JsonObject jsonLabels = json.getAsJsonObject("labels");
    @javax.annotation.Nonnull final Map<UUID, Layer> source_layersByNodeId = new HashMap<>();
    @javax.annotation.Nonnull final Map<UUID, Layer> source_layersByLayerId = new HashMap<>();
    for (@javax.annotation.Nonnull final Entry<String, JsonElement> e : jsonLayers.entrySet()) {
      @javax.annotation.Nonnull Layer value = Layer.fromJson(e.getValue().getAsJsonObject(), rs);
      source_layersByLayerId.put(UUID.fromString(e.getKey()), value);
    }
    for (@javax.annotation.Nonnull final Entry<String, JsonElement> e : jsonNodes.entrySet()) {
      @javax.annotation.Nonnull final UUID nodeId = UUID.fromString(e.getKey());
      @javax.annotation.Nonnull final UUID layerId = UUID.fromString(e.getValue().getAsString());
      final Layer layer = source_layersByLayerId.get(layerId);
      assert null != layer;
      source_layersByNodeId.put(nodeId, layer);
    }
    @javax.annotation.Nonnull final LinkedHashMap<String, UUID> labels = new LinkedHashMap<>();
    for (@javax.annotation.Nonnull final Entry<String, JsonElement> e : jsonLabels.entrySet()) {
      labels.put(e.getKey(), UUID.fromString(e.getValue().getAsString()));
    }
    @javax.annotation.Nonnull final Map<UUID, List<UUID>> deserializedLinks = new HashMap<>();
    for (@javax.annotation.Nonnull final Entry<String, JsonElement> e : jsonLinks.entrySet()) {
      @javax.annotation.Nonnull final ArrayList<UUID> linkList = new ArrayList<>();
      for (@javax.annotation.Nonnull final JsonElement linkItem : e.getValue().getAsJsonArray()) {
        linkList.add(UUID.fromString(linkItem.getAsString()));
      }
      deserializedLinks.put(UUID.fromString(e.getKey()), linkList);
    }
    for (final UUID key : labels.values()) {
      initLinks(deserializedLinks, source_layersByNodeId, key);
    }
    @javax.annotation.Nonnull final UUID head = UUID.fromString(json.getAsJsonPrimitive("head").getAsString());
    initLinks(deserializedLinks, source_layersByNodeId, head);
    source_layersByLayerId.values().forEach(x -> x.freeRef());
    this.labels.putAll(labels);
    assertConsistent();
  }
  
  /**
   * Add dag node.
   *
   * @param nextHead the next head
   * @param head     the head
   * @return the dag node
   */
  @javax.annotation.Nullable
  public InnerNode add(@javax.annotation.Nonnull final Layer nextHead, final DAGNode... head) {
    return add(null, nextHead, head);
  }
  
  /**
   * Wrap dag node.
   *
   * @param nextHead the next head
   * @param head     the head
   * @return the dag node
   */
  @javax.annotation.Nullable
  public InnerNode wrap(@javax.annotation.Nonnull final Layer nextHead, final DAGNode... head) {
    InnerNode add = add(null, nextHead, head);
    nextHead.freeRef();
    return add;
  }
  
  /**
   * Add dag node.
   *
   * @param label the label
   * @param layer the layer
   * @param head  the head
   * @return the dag node
   */
  public InnerNode add(@Nullable final String label, @javax.annotation.Nonnull final Layer layer, final DAGNode... head) {
    assertAlive();
    assertConsistent();
    assert null != getInput();
    @javax.annotation.Nonnull final InnerNode node = new InnerNode(this, layer, head);
    synchronized (layersById) {
      if (!layersById.containsKey(layer.getId())) {
        Layer replaced = layersById.put(layer.getId(), layer);
        layer.addRef();
        if (null != replaced) replaced.freeRef();
      }
    }
    DAGNode replaced = nodesById.put(node.getId(), node);
    if (null != replaced) replaced.freeRef();
    if (null != label) {
      labels.put(label, node.getId());
    }
    assertConsistent();
    return node;
  }
  
  @Override
  protected void _free() {
    super._free();
    this.layersById.values().forEach(ReferenceCounting::freeRef);
    this.nodesById.values().forEach(ReferenceCounting::freeRef);
    this.inputNodes.values().forEach(ReferenceCounting::freeRef);
  }
  
  /**
   * Add input nn layer.
   *
   * @return the nn layer
   */
  @javax.annotation.Nonnull
  public Layer addInput() {
    @javax.annotation.Nonnull final UUID key = UUID.randomUUID();
    inputHandles.add(key);
    InputNode replaced = inputNodes.put(key, new InputNode(this, key));
    if (null != replaced) replaced.freeRef();
    return this;
  }
  
  /**
   * Assert consistent boolean.
   *
   * @return the boolean
   */
  protected boolean assertConsistent() {
    assert null != getInput();
    for (@javax.annotation.Nonnull final Entry<String, UUID> e : labels.entrySet()) {
      assert nodesById.containsKey(e.getValue());
    }
    for (@javax.annotation.Nonnull final Entry<UUID, DAGNode> e : nodesById.entrySet()) {
      @Nullable final Layer layer = e.getValue().getLayer();
      assert layersById.containsKey(layer.getId());
      assert layersById.get(layer.getId()) == layer;
    }
    return true;
  }
  
  /**
   * Attach.
   *
   * @param obj the obj
   */
  public void attach(@javax.annotation.Nonnull final MonitoredObject obj) {
    visitLayers(layer -> {
      if (layer instanceof MonitoredItem) {
        obj.addObj(layer.getName(), (MonitoredItem) layer);
      }
    });
  }
  
  /**
   * Build exe ctx graph evaluation context.
   *
   * @param inputs the inputs
   * @return the graph evaluation context
   */
  @javax.annotation.Nonnull
  public GraphEvaluationContext buildExeCtx(@javax.annotation.Nonnull final Result... inputs) {
    assert inputs.length == inputHandles.size() : inputs.length + " != " + inputHandles.size();
    @javax.annotation.Nonnull final GraphEvaluationContext context = new GraphEvaluationContext();
    for (int i = 0; i < inputs.length; i++) {
      UUID key = inputHandles.get(i);
      Result input = inputs[i];
      if (!context.calculated.containsKey(key)) {
        input.getData().addRef();
        context.calculated.put(key, new Singleton<CountingResult>().set(new CountingResult(input)));
      }
    }
    context.expectedCounts.putAll(getNodes().stream().flatMap(t -> {
      return Arrays.stream(t.getInputs()).map(n -> n.getId());
    }).filter(x -> !inputHandles.contains(x)).collect(Collectors.groupingBy(x -> x, Collectors.counting())));
    return context;
  }
  
  @javax.annotation.Nonnull
  @Override
  public DAGNetwork copy(SerialPrecision precision) {
    return (DAGNetwork) super.copy(precision);
  }
  
  @javax.annotation.Nullable
  @Override
  public Result eval(final Result... input) {
    assertAlive();
    @javax.annotation.Nonnull GraphEvaluationContext buildExeCtx = buildExeCtx(input);
    @javax.annotation.Nullable Result result;
    try {
      result = getHead().get(buildExeCtx);
    } finally {
      buildExeCtx.freeRef();
    }
    return result;
  }
  
  /**
   * Gets by label.
   *
   * @param key the key
   * @return the by label
   */
  public DAGNode getByLabel(final String key) {
    return nodesById.get(labels.get(key));
  }
  
  /**
   * Gets by name.
   *
   * @param <T>  the type parameter
   * @param name the name
   * @return the by name
   */
  @Nullable
  @SuppressWarnings("unchecked")
  public <T extends Layer> T getByName(@Nullable final String name) {
    if (null == name) return null;
    @javax.annotation.Nonnull final AtomicReference<Layer> result = new AtomicReference<>();
    visitLayers(n -> {
      if (name.equals(n.getName())) {
        result.set(n);
      }
    });
    return (T) result.get();
  }
  
  /**
   * Gets child node.
   *
   * @param id the id
   * @return the child node
   */
  public DAGNode getChildNode(final UUID id) {
    if (nodesById.containsKey(id)) {
      return nodesById.get(id);
    }
    return nodesById.values().stream().map(x -> x.getLayer())
      .filter(x -> x instanceof DAGNetwork)
      .map(x -> ((DAGNetwork) x).getChildNode(id)).findAny().orElse(null);
  }
  
  @Override
  public List<Layer> getChildren() {
    return layersById.values().stream().flatMap(l -> l.getChildren().stream()).distinct().sorted(Comparator.comparing(l -> l.getId().toString())).collect(Collectors.toList());
  }
  
  private DAGNode[] getDependencies(@javax.annotation.Nonnull final Map<UUID, List<UUID>> deserializedLinks, final UUID e) {
    final List<UUID> links = deserializedLinks.get(e);
    if (null == links) return new DAGNode[]{};
    return links.stream().map(id -> getNode(id)).toArray(i -> new DAGNode[i]);
  }
  
  /**
   * Gets head.
   *
   * @return the head
   */
  @Nullable
  public abstract DAGNode getHead();
  
  /**
   * Gets input.
   *
   * @return the input
   */
  @javax.annotation.Nonnull
  public List<DAGNode> getInput() {
    @javax.annotation.Nonnull final ArrayList<DAGNode> list = new ArrayList<>();
    for (final UUID key : inputHandles) {
      list.add(inputNodes.get(key));
    }
    return list;
  }
  
  /**
   * Gets input.
   *
   * @param index the index
   * @return the input
   */
  public DAGNode getInput(final int index) {
    final DAGNode input = inputNodes.get(inputHandles.get(index));
    assert null != input;
    return input;
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    @javax.annotation.Nonnull final JsonArray inputs = new JsonArray();
    json.add("inputs", inputs);
    inputHandles.forEach(uuid -> inputs.add(new JsonPrimitive(uuid.toString())));
    @javax.annotation.Nonnull final JsonObject layerMap = new JsonObject();
    @javax.annotation.Nonnull final JsonObject nodeMap = new JsonObject();
    @javax.annotation.Nonnull final JsonObject links = new JsonObject();
    nodesById.values().forEach(node -> {
      @javax.annotation.Nonnull final JsonArray linkArray = new JsonArray();
      Arrays.stream(node.getInputs()).forEach((@javax.annotation.Nonnull final DAGNode input) -> linkArray.add(new JsonPrimitive(input.getId().toString())));
      @Nullable final Layer layer = node.getLayer();
      @javax.annotation.Nonnull final String nodeId = node.getId().toString();
      final String layerId = layer.getId().toString();
      nodeMap.addProperty(nodeId, layerId);
      layerMap.add(layerId, layer.getJson(resources, dataSerializer));
      links.add(nodeId, linkArray);
    });
    json.add("nodes", nodeMap);
    json.add("layers", layerMap);
    json.add("links", links);
    @javax.annotation.Nonnull final JsonObject labels = new JsonObject();
    this.labels.forEach((k, v) -> {
      labels.addProperty(k.toString(), v.toString());
    });
    json.add("labels", labels);
    json.addProperty("head", getHead().getId().toString());
    return json;
  }
  
  /**
   * Gets layer.
   *
   * @return the layer
   */
  @javax.annotation.Nonnull
  public Layer getLayer() {
    return this;
  }
  
  private DAGNode getNode(final UUID id) {
    DAGNode returnValue = nodesById.get(id);
    if (null == returnValue) {
      returnValue = inputNodes.get(id);
    }
    return returnValue;
  }
  
  /**
   * Gets nodes.
   *
   * @return the nodes
   */
  public List<DAGNode> getNodes() {
    return Stream.concat(
      nodesById.values().stream(),
      getInput().stream()
    ).collect(Collectors.toList());
  }
  
  private synchronized void initLinks(@javax.annotation.Nonnull final Map<UUID, List<UUID>> nodeLinks, @javax.annotation.Nonnull final Map<UUID, Layer> layersByNodeId, final UUID newNodeId) {
    if (layersById.containsKey(newNodeId)) return;
    if (inputNodes.containsKey(newNodeId)) return;
    final Layer layer = layersByNodeId.get(newNodeId);
    if (layer == null) {
      throw new IllegalArgumentException(String.format("%s is linked to but not defined", newNodeId));
    }
    final List<UUID> links = nodeLinks.get(newNodeId);
    if (null != links) {
      for (final UUID link : links) {
        initLinks(nodeLinks, layersByNodeId, link);
      }
    }
    assertConsistent();
    final DAGNode[] dependencies = getDependencies(nodeLinks, newNodeId);
    @javax.annotation.Nonnull final InnerNode node = new InnerNode(this, layer, newNodeId, dependencies);
    if (!layersById.containsKey(layer.getId())) {
      Layer replaced = layersById.put(layer.getId(), layer);
      layer.addRef();
      if (null != replaced) replaced.freeRef();
    }
    DAGNode replaced = nodesById.put(node.getId(), node);
    if (null != replaced) replaced.freeRef();
    assertConsistent();
  }
  
  /**
   * Remove last input nn layer.
   *
   * @return the nn layer
   */
  @javax.annotation.Nonnull
  public Layer removeLastInput() {
    final int index = inputHandles.size() - 1;
    final UUID key = inputHandles.remove(index);
    InputNode remove = inputNodes.remove(key);
    if (null != remove) remove.freeRef();
    return this;
  }
  
  /**
   * Reset.
   */
  public synchronized void reset() {
    layersById.values().forEach(x -> x.freeRef());
    layersById.clear();
    nodesById.values().forEach(x -> x.freeRef());
    nodesById.clear();
    labels.clear();
  }
  
  @javax.annotation.Nonnull
  @Override
  public DAGNetwork setFrozen(final boolean frozen) {
    super.setFrozen(frozen);
    if (null == layersById) throw new IllegalStateException();
    visitLayers(layer -> layer.setFrozen(frozen));
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return getChildren().stream().flatMap(l -> l.state().stream()).distinct().collect(Collectors.toList());
  }
  
  /**
   * Visit layers.
   *
   * @param visitor the visitor
   */
  public void visitLayers(@javax.annotation.Nonnull final Consumer<Layer> visitor) {
    layersById.values().forEach(layer -> {
      if (layer instanceof DAGNetwork) {
        ((DAGNetwork) layer).visitLayers(visitor);
      }
      if (layer instanceof WrapperLayer) {
        visitor.accept(((WrapperLayer) layer).getInner());
      }
      visitor.accept(layer);
    });
  }
  
  /**
   * Visit nodes.
   *
   * @param visitor the visitor
   */
  public void visitNodes(@javax.annotation.Nonnull final Consumer<DAGNode> visitor) {
    nodesById.values().forEach(node -> {
      Layer layer = node.getLayer();
      while (layer instanceof WrapperLayer) {
        layer = ((WrapperLayer) layer).getInner();
      }
      if (layer instanceof DAGNetwork) {
        ((DAGNetwork) layer).visitNodes(visitor);
      }
      visitor.accept(node);
    });
  }
  
}
