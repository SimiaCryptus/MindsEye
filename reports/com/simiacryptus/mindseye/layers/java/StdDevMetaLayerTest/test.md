# StdDevMetaLayer
## StdDevMetaLayerTest
### Json Serialization
Code from [LayerTestBase.java:75](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L75) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.StdDevMetaLayer",
      "id": "b385277b-2d2d-42fe-8250-210c0000ed6f",
      "isFrozen": false,
      "name": "StdDevMetaLayer/b385277b-2d2d-42fe-8250-210c0000ed6f",
      "inputs": [
        "9556312c-154a-4f07-b00e-d4a69c0e9405"
      ],
      "nodes": {
        "0d9c784a-7782-4b3f-9f55-c01b2c12a161": "b385277b-2d2d-42fe-8250-210c0000ed73",
        "7bc83e2b-4703-4e97-9287-7ab974fff7a1": "b385277b-2d2d-42fe-8250-210c0000ed72",
        "16cee564-481f-4779-89e8-4ce3be1d3ef0": "b385277b-2d2d-42fe-8250-210c0000ed76",
        "2b355ab0-a466-4edf-85dc-f4b7006b687c": "b385277b-2d2d-42fe-8250-210c0000ed75",
        "778bbc53-00ef-45c6-b694-d84e36567fa7": "b385277b-2d2d-42fe-8250-210c0000ed74",
        "b573b6e6-f954-49a2-8052-fd328c1929b0": "b385277b-2d2d-42fe-8250-210c0000ed71",
        "91878ab3-8d46-4817-bee9-1177f475eb34": "b385277b-2d2d-42fe-8250-210c0000ed70"
      },
      "layers": {
        "b385277b-2d2d-42fe-8250-210c0000ed73": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "b385277b-2d2d-42fe-8250-210c0000ed73",
          "isFrozen": true,
          "name": "SqActivationLayer/b385277b-2d2d-42fe-8250-210c0000ed73"
        },
        "b385277b-2d2d-42fe-8250-210c0000ed72": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "b385277b-2d2d-42fe-8250-210c0000ed72",
          "isFrozen": false,
          "name": "AvgMetaLayer/b385277b-2d2d-42fe-8250-210c0000ed72"
        },
        "b385277b-2d2d-42fe-8250-210c0000ed76": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "b385277b-2d2d-42fe-8250-210c0000ed76",
          "isFrozen": false,
          "name": "AvgMetaLayer/b385277b-2d2d-42fe-8250-210c0000ed76"
        },
        "b385277b-2d2d-42fe-8250-210c0000ed75": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "b385277b-2d2d-42fe-8250-210c0000ed75",
          "isFrozen": true,
          "name": "SqActivationLayer/b385277b-2d2d-42fe-8250-210c0000ed75"
        },
        "b385277b-2d2d-42fe-8250-210c0000ed74": {
          "class": "com.simiacryptus.mindseye.layers.java.LinearActivationLayer",
          "id": "b385277b-2d2d-42fe-8250-210c0000ed74",
          "isFrozen": false,
          "name": "LinearActivationLayer/b385277b-2d2d-42fe-8250-210c0000ed74",
          "weights": {
            "dimensions": [
              2
            ],
            "data": [
              -1.0,
              0.0
            ]
          }
        },
        "b385277b-2d2d-42fe-8250-210c0000ed71": {
          "class": "com.simiacryptus.mindseye.layers.java.SumInputsLayer",
          "id": "b385277b-2d2d-42fe-8250-210c0000ed71",
          "isFrozen": false,
          "name": "SumInputsLayer/b385277b-2d2d-42fe-8250-210c0000ed71"
        },
        "b385277b-2d2d-42fe-8250-210c0000ed70": {
          "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
          "id": "b385277b-2d2d-42fe-8250-210c0000ed70",
          "isFrozen": false,
          "name": "NthPowerActivationLayer/b385277b-2d2d-42fe-8250-210c0000ed70",
          "power": 0.5
        }
      },
      "links": {
        "0d9c784a-7782-4b3f-9f55-c01b2c12a161": [
          "9556312c-154a-4f07-b00e-d4a69c0e9405"
        ],
        "7bc83e2b-4703-4e97-9287-7ab974fff7a1": [
          "0d9c784a-7782-4b3f-9f55-c01b2c12a161"
        ],
        "16cee564-481f-4779-89e8-4ce3be1d3ef0": [
          "9556312c-154a-4f07-b00e-d4a69c0e9405"
        ],
        "2b355ab0-a466-4edf-85dc-f4b7006b687c": [
          "16cee564-481f-4779-89e8-4ce3be1d3ef0"
        ],
        "778bbc53-00ef-45c6-b694-d84e36567fa7": [
          "2b355ab0-a466-4edf-85dc-f4b7006b687c"
        ],
        "b573b6e6-f954-49a2-8052-fd328c1929b0": [
          "7bc83e2b-4703-4e97-9287-7ab974fff7a1",
          "778bbc53-00ef-45c6-b694-d84e36567fa7"
        ],
        "91878ab3-8d46-4817-bee9-1177f475eb34": [
          "b573b6e6-f954-49a2-8052-fd328c1929b0"
        ]
      },
      "labels": {},
      "head": "91878ab3-8d46-4817-bee9-1177f475eb34"
    }
```



### Network Diagram
Code from [LayerTestBase.java:86](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L86) executed in 0.20 seconds: 
```java
    return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.1.png)



### Example Input/Output Pair
Code from [LayerTestBase.java:112](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L112) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[ -1.984, 1.26, -1.344 ]]
    --------------------
    Output: 
    [ 0.0, 0.0, 0.0 ]
```



### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (9#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 0.60 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.0888 +- 0.0454 [0.0655 - 1.7013]
    Learning performance: 0.0282 +- 0.0243 [0.0171 - 1.2397]
    
```

