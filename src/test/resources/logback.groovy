  appender("CONSOLE", ConsoleAppender) {
    encoder(PatternLayoutEncoder) {
      pattern = "%d{HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n"
    }
  }
appender("FILE", FileAppender) {
    file = "test.log"
    append = true
    encoder(PatternLayoutEncoder) {
      pattern = "%d{HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n"
    }
  }
  root(DEBUG, ["CONSOLE", "FILE"])
  