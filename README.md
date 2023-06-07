# Проект команды Aboba :green_heart:
### Распознавание контента, непригодного для просмотра на рабочем месте.

Модераторы стримеров сталкиваются с серьезной проблемой запрещенного контента в интернете, включая порнографию и другие неприемлемые материалы.
В настоящий момент, чтобы стример мог посмотреть со зрителями фильм в прямом эфире, модератор должен заранее начать его отсматривать, предупреждать стримера **о каждом** запрещенном для показа кадре.

**Цель нашей работы** - нейросеть, способная распознавать, подходит ли для трансляции тот или иной контент.

<img src="https://user-images.githubusercontent.com/113666100/233092930-6b5b7850-8aa5-44b5-b0a7-ac56757f377c.jpg" width="400" height="400">

## Dataset :scroll:

Датасет, которым мы пользовались для обучения модели хранится в виде ссылок в локальном репозитории в связи с его содержанием:

https://bitbucket.org/desknsfw/123/src/master/

Пример фрагментов датасета:

<img src="https://github.com/CheesyPitsa/Lynnday/assets/113666100/ef924733-d98e-4134-8b45-5b4a90b3323d" height="600">

## Live-demo :tv:

https://github.com/CheesyPitsa/Lynnday/assets/113666100/1b9dacbe-a388-4957-b5d5-7a8218eccc97

## Структура модели :microscope:

```python
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(2048,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
```

<img src="https://github.com/CheesyPitsa/Lynnday/assets/113666100/d886df52-9138-40ca-8d0a-c7a7e5639c61" height="400">
