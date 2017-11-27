# StdDevMetaLayer
## StdDevMetaLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "id": "0910987d-3688-428c-a892-e2c400000eae",
      "isFrozen": false,
      "name": "StdDevMetaLayer/0910987d-3688-428c-a892-e2c400000eae",
      "inputs": [
        "abba68e5-1dd9-4e37-8924-9fe67e45ab9f"
      ],
      "nodes": {
        "f476951a-0ddb-4fc0-a567-c5aa7adaa28e": "0910987d-3688-428c-a892-e2c400000eb2",
        "7f7ec226-7891-49e6-9d9d-d78cfb617d8e": "0910987d-3688-428c-a892-e2c400000eb1",
        "49d4eb12-2a96-4b14-8c66-1425e56fef9b": "0910987d-3688-428c-a892-e2c400000eb5",
        "f93f43e0-63cd-456b-8c57-19f247f6c30c": "0910987d-3688-428c-a892-e2c400000eb4",
        "70d534aa-248d-42df-808c-4110d11aafad": "0910987d-3688-428c-a892-e2c400000eb3",
        "0954c6ec-c485-497d-bfab-a247c9a9c777": "0910987d-3688-428c-a892-e2c400000eb0",
        "d5329a01-835e-49c8-a973-caa0f21c8aa1": "0910987d-3688-428c-a892-e2c400000eaf"
      },
      "layers": {
        "0910987d-3688-428c-a892-e2c400000eb2": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "0910987d-3688-428c-a892-e2c400000eb2",
          "isFrozen": true,
          "name": "SqActivationLayer/0910987d-3688-428c-a892-e2c400000eb2"
        },
        "0910987d-3688-428c-a892-e2c400000eb1": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "0910987d-3688-428c-a892-e2c400000eb1",
          "isFrozen": false,
          "name": "AvgMetaLayer/0910987d-3688-428c-a892-e2c400000eb1"
        },
        "0910987d-3688-428c-a892-e2c400000eb5": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "0910987d-3688-428c-a892-e2c400000eb5",
          "isFrozen": false,
          "name": "AvgMetaLayer/0910987d-3688-428c-a892-e2c400000eb5"
        },
        "0910987d-3688-428c-a892-e2c400000eb4": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "0910987d-3688-428c-a892-e2c400000eb4",
          "isFrozen": true,
          "name": "SqActivationLayer/0910987d-3688-428c-a892-e2c400000eb4"
        },
        "0910987d-3688-428c-a892-e2c400000eb3": {
          "class": "com.simiacryptus.mindseye.layers.java.LinearActivationLayer",
          "id": "0910987d-3688-428c-a892-e2c400000eb3",
          "isFrozen": false,
          "name": "LinearActivationLayer/0910987d-3688-428c-a892-e2c400000eb3",
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
        "0910987d-3688-428c-a892-e2c400000eb0": {
          "class": "com.simiacryptus.mindseye.layers.java.SumInputsLayer",
          "id": "0910987d-3688-428c-a892-e2c400000eb0",
          "isFrozen": false,
          "name": "SumInputsLayer/0910987d-3688-428c-a892-e2c400000eb0"
        },
        "0910987d-3688-428c-a892-e2c400000eaf": {
          "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
          "id": "0910987d-3688-428c-a892-e2c400000eaf",
          "isFrozen": false,
          "name": "NthPowerActivationLayer/0910987d-3688-428c-a892-e2c400000eaf",
          "power": 0.5
        }
      },
      "links": {
        "f476951a-0ddb-4fc0-a567-c5aa7adaa28e": [
          "abba68e5-1dd9-4e37-8924-9fe67e45ab9f"
        ],
        "7f7ec226-7891-49e6-9d9d-d78cfb617d8e": [
          "f476951a-0ddb-4fc0-a567-c5aa7adaa28e"
        ],
        "49d4eb12-2a96-4b14-8c66-1425e56fef9b": [
          "abba68e5-1dd9-4e37-8924-9fe67e45ab9f"
        ],
        "f93f43e0-63cd-456b-8c57-19f247f6c30c": [
          "49d4eb12-2a96-4b14-8c66-1425e56fef9b"
        ],
        "70d534aa-248d-42df-808c-4110d11aafad": [
          "f93f43e0-63cd-456b-8c57-19f247f6c30c"
        ],
        "0954c6ec-c485-497d-bfab-a247c9a9c777": [
          "7f7ec226-7891-49e6-9d9d-d78cfb617d8e",
          "70d534aa-248d-42df-808c-4110d11aafad"
        ],
        "d5329a01-835e-49c8-a973-caa0f21c8aa1": [
          "0954c6ec-c485-497d-bfab-a247c9a9c777"
        ]
      },
      "labels": {},
      "head": "d5329a01-835e-49c8-a973-caa0f21c8aa1"
    }
```



### Network Diagram
Code from [LayerTestBase.java:95](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L95) executed in 0.20 seconds: 
```java
    return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.1.png)



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    [[ 1.372, 1.46, -1.22 ]]
    --------------------
    Output: 
    [ 0.0, 0.0, 0.0 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
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
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.3402 +- 0.0444 [0.2308 - 0.5586]
    Learning performance: 0.1112 +- 0.0335 [0.0826 - 0.3420]
    
```

