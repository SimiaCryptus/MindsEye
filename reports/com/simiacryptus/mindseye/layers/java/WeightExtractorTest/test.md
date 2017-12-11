# WeightExtractor
## WeightExtractorTest
### Json Serialization
Code from [LayerTestBase.java:121](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.WeightExtractor",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede0000369b",
      "isFrozen": false,
      "name": "WeightExtractor/e2a3bda5-e7e7-4c05-aeb3-4ede0000369b",
      "innerId": "e2a3bda5-e7e7-4c05-aeb3-4ede0000369a",
      "index": 0
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:159](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L159) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[  ]]
    --------------------
    Output: 
    [ 0.2768048117735981, -0.9102150198432373, -0.1662372788527983, -0.727717491111909, 0.5653129481228371, 0.3318441077637774, 0.7545355055068854, 0.39345657138778384, -0.0074336805964254115 ]
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1274 +- 0.0144 [0.1083 - 0.2166]
    Learning performance: 0.0116 +- 0.0077 [0.0057 - 0.0656]
    
```

