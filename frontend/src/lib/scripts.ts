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
    body: `Hi. My name is Alex, and I live near the park on the north side of town. This week I have been walking there after work, usually once the day has cooled down. The path goes past the pond, then up a small hill, and comes out by the library.

Yesterday I went a little later than usual. Two ducks were arguing over a crust of bread near the water. A very serious dog in a yellow coat walked by with its person, and both of them looked like they had somewhere important to be. Farther along, a red kite with a long blue tail was stuck in a tree. A kid was describing the kite to anyone who would listen.

If you have a minute, I can tell you the rest. I sat on the bench by the hill and watched the light change on the water. Someone nearby was reading out loud, quietly, just to themselves. It made the whole place feel unhurried. I stayed until the lamps came on, then walked home the long way, past the bakery that still had the door open.`,
  },
  {
    id: "plan",
    title: "Making a plan",
    body: `I was thinking we could meet on Thursday around four. The cafe on Pine Street has a table by the window, and they keep the good cake until late afternoon. I can get there a little early and save us seats. If Thursday is tight, Friday morning works too. Just tell me which one is easier for you.

If we do Thursday, I will take the bus that stops on the corner by the bookshop. It is usually on time, but I will leave a little extra room in case the traffic is slow. I do not need anything from the shop unless you want a book picked up on the way. Send me the title and I will look for it.

Friday would be quieter. We could sit longer, and the cafe opens at eight. I have nothing else that morning until a call at eleven. Either day is fine with me. I am looking forward to it, and there is no rush to decide tonight. A message tomorrow is plenty.`,
  },
  {
    id: "call",
    title: "A short call",
    body: `Hello, this is Jordan. I am calling about the appointment on Tuesday at ten thirty. I can still make it, and I will be there a few minutes early. If that time has changed, a message is fine. My number is the one I called from. Thank you for checking.

I wanted to mention two small things before Tuesday. The first is the form you sent last week. I filled it out and I can bring a paper copy, or I can send it back this evening, whichever is easier for the office. The second is the door. Last time I was not sure whether to use the front entrance or the side gate, so I waited outside longer than I meant to.

If someone can point me to the right door, that would help. I will ask at the desk if the building looks busy. I do not need a long appointment. Ten or fifteen minutes should be enough to go over the form and ask one question about the date of the next visit. Thanks again for taking the call.`,
  },
  {
    id: "yesterday",
    title: "Yesterday",
    body: `Yesterday the train was late, so I waited on the platform with a book. Rain started, then stopped, then started again. When the train came I found a seat by the door and watched the stations go by. Maple, Harbor, then the long bridge. I got off at mine, bought oranges on the way home, and put the kettle on.

The platform was more crowded than I expected for a weekday afternoon. A person in a green jacket was explaining the delay to a friend, and they both laughed when the board changed again. I read the same page twice because I kept looking up. None of it felt urgent. The book could wait, and so could I.

On the train I stood for the first few minutes, then a seat opened by the window. The glass was fogged at the corners. Past the bridge the rain eased, and the light on the water was bright enough that people put their phones down and just looked. At my stop I stepped off carefully, because the platform was wet. The shop on the corner had oranges in a crate out front. I bought four, walked the last two blocks, and listened to the kettle as it warmed.`,
  },
  {
    id: "kitchen",
    title: "In the kitchen",
    body: `I am standing in the kitchen. The kettle is on, and the window is open a little. There is bread on the board, two mugs by the sink, and a list on the fridge that says milk, oranges, and rice. In a minute I will pour the tea, sit down with it, and decide what to make for dinner.

The light in here is best in the late afternoon. It comes across the table and lands on the blue bowl, the one with a chip on the rim that I keep meaning to replace. I have not replaced it, and I probably will not. It still holds fruit just fine. Today the bowl has two apples and a lemon. The lemon was a gift from a neighbor who grew it in a pot by their stairs.

Dinner can be simple. Rice, something green, and the rest of the bread if it is still soft. I might toast the bread and put butter on it while the rice cooks. There is no one else to please tonight, which makes the choosing easier. I will eat at the table instead of standing at the counter, and I will leave the window open until the room gets cool.`,
  },
];
