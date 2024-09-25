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
## История
Я много играл в игру [Сколькиугодномерный лабиринт](https://kirillkirin.ru/4d/4d.html), конкретно - проходил четырёхмерные лабиринты. Сохранял четырехмерные карты и старался их запоминать. Каждая такая карта является остовным деревом (spanning tree) гиперкуба. Я заметил, что карту легче запомнить, если найти в ней куб, который можно обойти, не выходя из него в другие области лабиринта, то есть не используя четвёртое измерение. Другими словами, если найти куб, у которого есть свой остовный граф. Дальше я задался вопросом - в каждом ли остовном графе гиперкуба должен быть подграф-остовный граф куба? По крайней мере, другие мне не попадались. И поэтому я написал скрипт для поиска карты, в которой не будет такого подграфа. Правда, в процессе написания стало и так понятно, что такая карта существует и её можно нарисовать самому. Но найти все такие карты самому - уже не так просто.  


## Описание
Инструмент hypertools предназначен для создания и редактирования карт для игры [Сколькиугодномерный лабиринт](https://kirillkirin.ru/4d/4d.html). В данном  проекте, однако, поддерживаются только карты на четырёхмерном гиперкубе со стороной 1.
Картой для игры служит остовное дерево (spanning tree) гиперкуба.

## Возможности hypertools
Помимо графического отображения карты, загрузки и выгрузки, инструмент позволяет:
* Преобразование карты (повороты/отражения гиперкуба)
* Распутывание карты (приведение к виду с наименьшим числом пересечений на плоской проекции)
* Поиск карт, в которых нельзя обойти ни один куб (трёхмерную грань гиперкуба), не выходя из него.

Каждой возможной карте соответствует целочисленный индекс, не превышающий 2 в степени 32 (но не каждому такой целому числу соответствует карта). 

## [Как загрузить карту в игру](load.md)

