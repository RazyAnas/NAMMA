package namma.core

/** What the app says, from docs/CAPTURE_GUIDE.md (SplatTour's method + what broke on our own videos). */
object Guide {
    /** Ticked before recording can start. */
    val checklist = listOf(
        "Lights on, all of them, and keep them the same until the end",
        "Doors as you want them shown; nobody walks through while you film",
        "Plain walls: tape up a poster or put a chair or plant near them",
        "Start at the entrance. One video per floor, never cut it",
        "Phone upright (the app picks the widest lens itself)",
    )

    const val LOCK = "Point at the brightest wall and tap LOCK: exposure, colour and focus stay fixed for the whole video"

    class Pass(val title: String, val say: String, val tip: String)

    /** The passes in every room (one take). */
    val passes = listOf(
        Pass("Lap 1: along the walls", "Lap one. Walk along the walls, about one metre away, looking straight ahead.",
            "About 1 m from the walls, phone at chest height, looking straight ahead"),
        Pass("Lap 2: the other way", "Lap two. The same lap the other way round.",
            "Same lap, the other direction. Show big furniture from both sides"),
        Pass("Corner to corner", "Now cross the room corner to corner, both diagonals.",
            "Cross along both diagonals, slowly"),
        Pass("Next room or stop", "Walk through the doorway slowly, through the middle. Tap next room, or stop if this was the last one.",
            "Doorways: slowly through the middle, once in and once out"),
    )

    /** Always on screen while recording. */
    val rules = listOf(
        "Turn slowly: 3 seconds for a quarter turn",
        "Aim at eye level, keep the floor out of the picture",
        "Never orbit around one spot; walk in lines",
        "Mirrors: pass them sideways",
    )
}
