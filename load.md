```lang-none
            /|              /|
          //||            //||
        //  ||          //  ||  
      //    ||        //    ||    
    //      ||      //      ||      
  //        ||    //        ||         
//          ||  //      /|  ||          /|
||          ||  ||    //||  ||        //||
||          \\  ||  //  ||  \\      //  ||
||            \\||//    ||    \\  //    ||
||              //      ||      //      ||
||            //||\\    ||    //  \\    ||
||          //  ||  \\  ||  //      \\  ||
||          ||  ||    \\||  ||        \\||
\=>         || >=/      \|  ||          \|
            ||              ||         
            ||              ||       
            ||              ||     
            ||              ||   
            ||______________|| 
            ``````````````````
                        
```
# Использование save-файлов

Чтобы использовать save-файл, генерируемый функцией `export_graph`, необходимо выполнить следующие действия:

1. Открыть в текстовом редакторе (например, Блокнот) файл с сохранением. Выделить всё его содержимое и скопировать в буфер: `Ctrl+A`, `Ctrl+C`.
2. Открыть страницу с [игрой](https://kirillkirin.ru/4d/4d.html).
2. Если раньше не использовали сохранения - зайти в `настройки` (слева внизу), потом `Сохранить 1`, потом `Сохранить 2`.
3. Открыть инструменты разработчика, например клавишей `F12` или `Ctrl+Shift+I`.
4. В верхней части панели перейти на вкладку `Application`.
5. В левой части панели в разделе `Storage` открыть ветку `Local Storage` и выделить содержащийся в ней раздел. На панели станет видна таблица с заголовками `key` и `value`.
6. В столбце `value` найти ячейку рядом с ячейкой `save1`. Двойным щелчком по ячейке получить доступ к редактированию его содержимого. Выделить всё содержимое и вставить данные из файла: `Ctrl+A`, `Ctrl+V`.
7. Закрыть страницу с игрой и открыть снова.
8. Зайти в настройки (слева внизу), сначала нажать `Загрузить 2`, потом `Загрузить 1` (бывает, что при первой загрузке после запуска загружается случайная карта).
9. Нажать клавишу `назад`.
10. Располагайте курсор мыши в той стороне, куда хотите пойти, и щёлкайте!

Чтобы получить файл с сохранением игры, в инструментах разработчика найдите интересующий вас слот сохранения (`save1`, `save2`, `save3`), скопируйте его содержимое и сохраните в текстовый файл с расширением `.txt`.

