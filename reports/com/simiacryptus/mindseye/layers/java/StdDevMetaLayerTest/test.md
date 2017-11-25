### Json Serialization
Code from [LayerTestBase.java:57](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L57) executed in 0.00 seconds: 
```java
    NNLayer layer = getLayer();
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
      "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dd73",
      "isFrozen": false,
      "name": "StdDevMetaLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dd73",
      "inputs": [
        "a4af45a6-95fa-47cb-9c15-932f54e8083f"
      ],
      "nodes": {
        "288444eb-6e9a-42de-a3cf-aeedf8d4eabf": "9d13704a-9a5a-4ecb-a687-5c7c0002dd77",
        "4348e0ed-c4c3-4c3b-9a0f-a63e5757c6a2": "9d13704a-9a5a-4ecb-a687-5c7c0002dd76",
        "3673e6fc-f265-4c68-9f3f-2e4f2b6b1108": "9d13704a-9a5a-4ecb-a687-5c7c0002dd7a",
        "b36a19d7-92e4-49b5-85a5-319380a9055e": "9d13704a-9a5a-4ecb-a687-5c7c0002dd79",
        "a11afba6-d8f9-4c62-9c08-d43c4a1329bf": "9d13704a-9a5a-4ecb-a687-5c7c0002dd78",
        "a98bde4c-4ad3-4008-a417-341a86128a8c": "9d13704a-9a5a-4ecb-a687-5c7c0002dd75",
        "2faeb5ab-ab3a-4098-a06f-cc692f338afd": "9d13704a-9a5a-4ecb-a687-5c7c0002dd74"
      },
      "layers": {
        "9d13704a-9a5a-4ecb-a687-5c7c0002dd77": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dd77",
          "isFrozen": true,
          "name": "SqActivationLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dd77"
        },
        "9d13704a-9a5a-4ecb-a687-5c7c0002dd76": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dd76",
          "isFrozen": false,
          "name": "AvgMetaLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dd76"
        },
        "9d13704a-9a5a-4ecb-a687-5c7c0002dd7a": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dd7a",
          "isFrozen": false,
          "name": "AvgMetaLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dd7a"
        },
        "9d13704a-9a5a-4ecb-a687-5c7c0002dd79": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dd79",
          "isFrozen": true,
          "name": "SqActivationLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dd79"
        },
        "9d13704a-9a5a-4ecb-a687-5c7c0002dd78": {
          "class": "com.simiacryptus.mindseye.layers.java.LinearActivationLayer",
          "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dd78",
          "isFrozen": false,
          "name": "LinearActivationLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dd78",
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
        "9d13704a-9a5a-4ecb-a687-5c7c0002dd75": {
          "class": "com.simiacryptus.mindseye.layers.java.SumInputsLayer",
          "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dd75",
          "isFrozen": false,
          "name": "SumInputsLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dd75"
        },
        "9d13704a-9a5a-4ecb-a687-5c7c0002dd74": {
          "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
          "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dd74",
          "isFrozen": false,
          "name": "NthPowerActivationLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dd74",
          "power": 0.5
        }
      },
      "links": {
        "288444eb-6e9a-42de-a3cf-aeedf8d4eabf": [
          "a4af45a6-95fa-47cb-9c15-932f54e8083f"
        ],
        "4348e0ed-c4c3-4c3b-9a0f-a63e5757c6a2": [
          "288444eb-6e9a-42de-a3cf-aeedf8d4eabf"
        ],
        "3673e6fc-f265-4c68-9f3f-2e4f2b6b1108": [
          "a4af45a6-95fa-47cb-9c15-932f54e8083f"
        ],
        "b36a19d7-92e4-49b5-85a5-319380a9055e": [
          "3673e6fc-f265-4c68-9f3f-2e4f2b6b1108"
        ],
        "a11afba6-d8f9-4c62-9c08-d43c4a1329bf": [
          "b36a19d7-92e4-49b5-85a5-319380a9055e"
        ],
        "a98bde4c-4ad3-4008-a417-341a86128a8c": [
          "4348e0ed-c4c3-4c3b-9a0f-a63e5757c6a2",
          "a11afba6-d8f9-4c62-9c08-d43c4a1329bf"
        ],
        "2faeb5ab-ab3a-4098-a06f-cc692f338afd": [
          "a98bde4c-4ad3-4008-a417-341a86128a8c"
        ]
      },
      "labels": {},
      "head": "2faeb5ab-ab3a-4098-a06f-cc692f338afd"
    }
```



### Differential Validation
Code from [LayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (9#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

### Performance
Code from [LayerTestBase.java:79](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L79) executed in 0.07 seconds: 
```java
    getPerformanceTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Forward performance: 3.5960 +- 2.9808 [2.0832 - 31.1909]
    Backward performance: 3.0007 +- 0.9323 [1.7754 - 7.8512]
    
```

### Reference Implementation
