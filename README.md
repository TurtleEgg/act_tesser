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
Инструмент hypertools предназначен для создания и редактирования карт для игры [Сколькиугодномерный лабиринт](https://kirillkirin.ru/4d/4d.html). В данном  проекте, однако, поддерживаются только карты на четырёхмерном гиперкубе со стороной 1.
Картой для игры служит остовное дерево (spanning tree) гиперкуба.

## Возможности hypertools
Помимо графического отображения карты, загрузки и выгрузки, инструмент позволяет:
* Преобразование карты (повороты/отражения гиперкуба)
* Распутывание карты (приведение к виду с наименьшим числом пересечений на плоской проекции)
* Поиск карт, в которых нельзя обойти ни один куб (трёхмерную грань гиперкуба), не выходя из него.

Каждой возможной карте соответствует целочисленный индекс, не превышающий 2 в степени 32 (но не каждому такой целому числу соответствует карта). 

## [Как загрузить карту в игру](load.md)

