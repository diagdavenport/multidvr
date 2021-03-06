% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/text.lms.R
\name{lms.test}
\alias{lms.test}
\title{Run the lms test on a textual outcome}
\usage{
lms.test(text, condition, data, iter = 1000, term.count.min = 10)
}
\arguments{
\item{text}{A character vector.}

\item{condition}{A binary vector.}
}
\value{
A list of stuff detailing how predictive \code{text} is for \code{condition}.
}
\description{
Run the lms test on a textual outcome
}
