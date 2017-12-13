# WeightExtractor
## WeightExtractorTest
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    if ((echo == null)) throw new AssertionError("Failed to deserialize");
    if ((layer == echo)) throw new AssertionError("Serialization did not copy");
    if ((!layer.equals(echo))) throw new AssertionError("Serialization not equal");
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.WeightExtractor",
      "id": "1f042bc3-57ce-4349-85e2-622de93a4c1a",
      "isFrozen": false,
      "name": "WeightExtractor/1f042bc3-57ce-4349-85e2-622de93a4c1a",
      "innerId": "cc3f22dd-4eec-4efa-8940-d1a96a50bf43",
      "index": 0
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n--------------------\nDerivative: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint(),
      Arrays.stream(eval.getDerivative()).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get());
```

Returns: 

```
    --------------------
    Input: 
    [[  ]]
    --------------------
    Output: 
    [ 0.09271630446520222, -0.09006385044726407, -0.5094230460745771, 0.0102773574506938, -0.8180370638876456, -0.3507616323525955, -0.8559885535020723, -0.34849976607860444, -0.7455292402903565 ]
    --------------------
    Derivative: 
    [  ]
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000031s +- 0.000012s [0.000021s - 0.000054s]
    Learning performance: 0.000005s +- 0.000002s [0.000002s - 0.000009s]
    
```

