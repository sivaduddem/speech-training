export interface ReadingScript {
  id: string;
  title: string;
  body: string;
}

/** Everyday passages to read aloud. They are not drills and they are not scored. */
export const SCRIPTS: ReadingScript[] = [
  {
    id: "hello",
    title: "Saying hello",
    body: "Hi. My name is Alex, and I live near the park. This week I have been walking there after work. The path goes past the pond, then up a small hill. If you have a minute, I can tell you what I saw yesterday: two ducks, a very serious dog in a yellow coat, and a kite stuck in a tree.",
  },
  {
    id: "plan",
    title: "Making a plan",
    body: "I was thinking we could meet on Thursday around four. The cafe on Pine Street has a table by the window, and they keep the good cake until late afternoon. I can get there a little early and save us seats. If Thursday is tight, Friday morning works too. Just tell me which one is easier for you.",
  },
  {
    id: "call",
    title: "A short call",
    body: "Hello, this is Jordan. I am calling about the appointment on Tuesday at ten thirty. I can still make it, and I will be there a few minutes early. If that time has changed, a message is fine. My number is the one I called from. Thank you for checking.",
  },
  {
    id: "yesterday",
    title: "Yesterday",
    body: "Yesterday the train was late, so I waited on the platform with a book. Rain started, then stopped, then started again. When the train came I found a seat by the door and watched the stations go by: Maple, Harbor, then the long bridge. I got off at mine, bought oranges on the way home, and put the kettle on.",
  },
  {
    id: "kitchen",
    title: "In the kitchen",
    body: "I am standing in the kitchen. The kettle is on, and the window is open a little. There is bread on the board, two mugs by the sink, and a list on the fridge that says milk, oranges, and rice. In a minute I will pour the tea, sit down with it, and decide what to make for dinner.",
  },
];
