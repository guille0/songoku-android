/*
Sudoku - a fast Java Sudoku game creation library.
Copyright (C) 2017-2018  Stephan Fuhrmann

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Library General Public
License as published by the Free Software Foundation; either
version 2 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Library General Public License for more details.

You should have received a copy of the GNU Library General Public
License along with this library; if not, write to the
Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
Boston, MA  02110-1301, USA.
*/
package de.sfuhrm.sudoku.output;

/**
 * Basic functionality for most text formatters.
 * @author Stephan Fuhrmann
 */
abstract class AbstractTextFormatter implements GameMatrixFormatter {
    /** The String used to display an empty or not filled cell.
     */
    private String unknownCellContentCharacter = ".";

    /** The line separator used.
     */
    private String lineSeparator = "\n";

    /** Gets the String to use for unknown/unset cells.
     * @return the unknown cell content String.
     */
    public final String getUnknownCellContentCharacter() {
        return unknownCellContentCharacter;
    }

    /** Sets the String to use for unknown/unset cells.
     * @param newUnknownCellContentCharacter the new value for unknown cells.
     */
    public final void setUnknownCellContentCharacter(
            final String newUnknownCellContentCharacter) {
        this.unknownCellContentCharacter = newUnknownCellContentCharacter;
    }

    /** Gets the line separator String to use.
     * @return the line separator String.
     */
    public final String getLineSeparator() {
        return lineSeparator;
    }

    /** Sets the line separator String to use.
     * @param newLineSeparator the new String to use as line separator.
     */
    public final void setLineSeparator(final String newLineSeparator) {
        this.lineSeparator = newLineSeparator;
    }

    @Override
    public String documentStart() {
        return "";
    }

    @Override
    public String documentEnd() {
        return "";
    }
}
