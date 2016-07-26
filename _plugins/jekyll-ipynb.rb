module Jekyll
  class IPythonNotebook < Converter
    safe true
    priority :low

    def matches(ext)
      ext =~ /^\.ipynbref$/i
    end

    def output_ext(ext)
      ".html"
    end

    def convert(content)
      `ipython nbconvert --to html --template basic --stdout \`pwd\`/_ipynbs/#{content}`
    end
  end
end